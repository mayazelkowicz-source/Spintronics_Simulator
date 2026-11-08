# Simulating an energy-efficient In-Memory Compute AI accelerator by mapping Binarized Neural Networks to Spintronic MTJ arrays

Group 111: Maya Zelkowicz, Sarit Mishiev
Advisor: Dr. Amir Capua, HUJI Spintronics Lab. Mentor: Daniella Har Shalom.

This repository contains a simulator for running a Binarized Neural Network (BNN) on virtual Magnetic Tunnel Junction (MTJ) crossbar arrays. Weights and activations are restricted to +1 and -1. The current version performs ideal integer inference, an analog MTJ device model, and statistical noise injection (resistance variation and bit flips), to study how device physics affects classification accuracy.

## Repository contents

- `main.py`: trains the BNN on MNIST with PyTorch and exports the weights and test vectors.
- `export_full_test.py`: exports the full 10,000-image test set from the existing weights, without retraining, and checks the first 100 samples against the committed test vectors.
- `layer1_weights.csv`, `layer2_weights.csv`: binarized weights, values +1/-1.
- `layer1_weights_bin.csv`, `layer2_weights_bin.csv`: the same weights in 0/1 encoding (1 = +1, 0 = -1).
- `test_inputs_pm1.csv`, `test_expected_logits.csv`, `test_labels.csv`: reference test vectors for the first 100 MNIST test images.
- `simulator_core/`: C++17 simulator (crossbar data structure, ideal integer forward pass, MTJ device model, Noise Injector, regression test and unit tests in `tests/`).
- `111_preliminary_report.md`, `111_preliminary_report.pdf`: Milestone 1 report.
- `PROGRESS_REPORT.md`: milestone summary.

## 1. Training the BNN (`main.py`)

Requirements: Python 3.9 or newer, PyTorch, torchvision, NumPy.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy
python3 main.py
```

The script performs the following steps:

1. Downloads MNIST into `./data/` (excluded from Git).
2. Trains a 784-256-10 binarized multilayer perceptron for 20 epochs (batch size 128, Adam optimizer, cosine learning-rate schedule, random seed 0). The sign function is trained with a straight-through estimator. The layers have no bias and no batch normalization, so each layer maps directly onto one crossbar array. A GPU is used if available.
3. Exports each layer's binarized weights in crossbar layout: rows are inputs (word lines) and columns are outputs (bit lines). Layer 1 is 784 x 256 and layer 2 is 256 x 10. Two files are written per layer:
   - `layerN_weights.csv`: values +1/-1
   - `layerN_weights_bin.csv`: values 0/1
4. Re-runs inference on the full test set with integer-only NumPy code using the exported files, and prints the resulting accuracy.
5. Writes the reference test vectors for the C++ simulator: 100 binarized test images, their expected output scores (logits) and their labels.

Note: running `main.py` overwrites the CSV files in the repository root. Exact weights may differ between machines (for example GPU vs. CPU). The test vectors are regenerated together with the weights, so the C++ test remains consistent.

## 2. Building and running the C++ simulator

Requirements: a C++17 compiler (g++ or clang++) and CMake 3.10 or newer.

```bash
cmake -S simulator_core -B simulator_core/build
cmake --build simulator_core/build
./simulator_core/build/bnn_sim
```

Run the binary from the repository root so it can find the CSV files. Alternative weight files can be passed as arguments: `bnn_sim <layer1.csv> <layer2.csv> [test_dir]`.

The program loads both weight matrices and runs ideal inference on the 100 reference images. It then compares every output score with `test_expected_logits.csv` and reports the classification accuracy.

It then runs the same network on the MTJ device model (`MTJParams.h`, `MTJCrossbar.h/.cpp`). Each weight is stored in a pair of MTJ cells on two bit lines: +1 puts the first cell in the Parallel (low resistance) state and the second in the Anti-Parallel (high resistance) state, and -1 does the opposite. Inputs are applied as +V_read or -V_read. The difference between the two bit-line currents is proportional to the integer column sum, so the input-dependent offset cancels. The sense step compares against half of one unit current (the current of a column sum of 1), which keeps sign(0) = +1 and exact ties in the final class identical to the integer model despite floating-point rounding. The values of V_read, R_P and TMR in `MTJParams.h` are placeholders from typical published values, to be replaced with lab data. Noise is off by default (see Section 4).

The exit code is 0 only when all integer outputs match exactly and the MTJ model predicts the same class as the integer model for every image.

Expected output (abridged):

```
MTJ device model (PLACEHOLDER values, not lab data):
  V_read = 0.1 V, R_P = 5000 ohm, TMR = 100%, R_AP = 10000 ohm
  Unit current (one P/AP pair at V_read) = 1e-05 A
  Physical MTJ devices (two per weight): Layer1 401408, Layer2 5120, total 406528

Ideal inference on 100 test samples:
  Logits matching golden reference: 100 / 100
  Classification accuracy:          96 / 100 (96%)
[PASS] Simulator is bit-exact with the golden reference.

Analog MTJ inference (noise off) on 100 test samples:
  Classes matching integer model:   100 / 100
  Classification accuracy:          96 / 100 (96%)
  ...
[PASS] Analog MTJ model predicts the same class as the integer model.
```

## 3. Full test-set verification

Run from the repository root, after building the simulator:

```bash
.venv/bin/python export_full_test.py
./simulator_core/build/bnn_sim layer1_weights_bin.csv layer2_weights_bin.csv full_test
```

The script downloads MNIST into `./data/` if needed and writes the test files to `full_test/` (excluded from Git). Result: logits match the reference on 10,000 of 10,000 images, with 94.30% accuracy in both Python and C++. The MTJ device model predicts the same class as the integer model on 10,000 of 10,000 images.

## 4. Noise injection

The Noise Injector (`NoiseInjector.h/.cpp`) changes the stored conductance of individual MTJ devices before inference. Each noise model implements the `NoiseModel` interface, so new models can be added without changing the crossbar or the sense logic. All models share one random generator (`std::mt19937_64`) seeded from the command line, so the same seed always gives the same result, on any compiler.

- **Resistance variation** (`--sigma S`): every device's resistance is multiplied by an independent lognormal factor exp(S * z), z ~ N(0, 1). S is the standard deviation of ln R; S = 0.2 is about 20% variation.
- **Bit flips** (`--p P`), with two modes (`--flip-mode`):
  - `cell` (default): each MTJ device flips between P and AP independently with probability P. One flipped device of a pair makes the weight contribute 0; both flipped invert the weight.
  - `weight`: each weight flips with probability P by flipping both of its devices, as in Hirtzlin et al. (2019).
  A flipped device keeps its own resistance variation, so its R_AP / R_P ratio stays equal to 1 + TMR.

```bash
./simulator_core/build/bnn_sim --sigma 0.1 --p 0.01 --seed 7
./simulator_core/build/bnn_sim layer1_weights_bin.csv layer2_weights_bin.csv full_test --p 0.05 --flip-mode weight
```

Positional arguments work as before and options can appear anywhere. Defaults: sigma 0, p 0, flip mode `cell`, seed 1. With sigma = 0 and p = 0 the conductances are unchanged and the output is the same as without options. When noise is on, the integer check still prints `[PASS]`, and the analog check is replaced by the noisy accuracy, the noise-free accuracy and the drop in percentage points.

## Testing

There is no automated CI. Before pushing changes, build the simulator and run:

```bash
./simulator_core/build/bnn_sim                                                         # both [PASS] lines
./simulator_core/build/bnn_sim layer1_weights_bin.csv layer2_weights_bin.csv full_test  # both [PASS] lines
ctest --test-dir simulator_core/build --output-on-failure                              # unit tests
```

The unit tests (`simulator_core/tests/noise_tests.cpp`) check that zero noise matches the integer model, that p = 1 negates every column sum in both flip modes, that one flipped device removes exactly its weight's contribution, seed reproducibility, the noise statistics, and that invalid parameters are rejected.

## Current status

- Done: BNN trained (94.30% MNIST test accuracy); weights and test vectors exported.
- Done: C++ crossbar data structure and ideal inference, matching the reference outputs exactly on all 10,000 MNIST test images.
- Done: noise-free MTJ device model (P/AP differential pairs, bit-line currents), matching the integer model's predicted class on all 10,000 test images.
- Done: Noise Injector with lognormal resistance variation and bit flips (per-device or whole-weight), seeded and unit-tested.
- Next: replace the placeholder device values with lab data, and sweep accuracy against noise over many seeds.
