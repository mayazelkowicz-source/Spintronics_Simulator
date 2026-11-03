# Simulating an energy-efficient In-Memory Compute AI accelerator by mapping Binarized Neural Networks to Spintronic MTJ arrays

Group 111: Maya Zelkowicz, Sarit Mishiev
Advisor: Dr. Amir Capua, HUJI Spintronics Lab. Mentor: Daniella Har Shalom.

This repository contains a simulator for running a Binarized Neural Network (BNN) on virtual Magnetic Tunnel Junction (MTJ) crossbar arrays. Weights and activations are restricted to +1 and -1. The current version performs ideal (noise-free) inference. Later stages will add MTJ device models and noise injection, to study how device physics affects classification accuracy.

## Repository contents

- `main.py`: trains the BNN on MNIST with PyTorch and exports the weights and test vectors.
- `export_full_test.py`: exports the full 10,000-image test set from the existing weights, without retraining, and checks the first 100 samples against the committed test vectors.
- `layer1_weights.csv`, `layer2_weights.csv`: binarized weights, values +1/-1.
- `layer1_weights_bin.csv`, `layer2_weights_bin.csv`: the same weights in 0/1 encoding (1 = +1, 0 = -1).
- `test_inputs_pm1.csv`, `test_expected_logits.csv`, `test_labels.csv`: reference test vectors for the first 100 MNIST test images.
- `simulator_core/`: C++17 simulator (crossbar data structure, ideal forward pass, regression test).
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

The program loads both weight matrices and runs ideal inference on the 100 reference images. It then compares every output score with `test_expected_logits.csv` and reports the classification accuracy. The exit code is 0 only when all outputs match exactly.

Expected output (abridged):

```
Ideal inference on 100 test samples:
  Logits matching golden reference: 100 / 100
  Classification accuracy:          96 / 100 (96%)
[PASS] Simulator is bit-exact with the golden reference.
```

## 3. Full test-set verification

Run from the repository root, after building the simulator:

```bash
.venv/bin/python export_full_test.py
./simulator_core/build/bnn_sim layer1_weights_bin.csv layer2_weights_bin.csv full_test
```

The script downloads MNIST into `./data/` if needed and writes the test files to `full_test/` (excluded from Git). Result: logits match the reference on 10,000 of 10,000 images, with 94.30% accuracy in both Python and C++.

## Testing

Regression testing is manual; there is no automated CI. Before pushing changes, build the simulator, run `bnn_sim` and confirm that it prints `[PASS]`.

## Current status

- Done: BNN trained (94.30% MNIST test accuracy); weights and test vectors exported.
- Done: C++ crossbar data structure and ideal inference, matching the reference outputs exactly on all 10,000 MNIST test images.
- Next: MTJ resistance and current model, and statistical noise injection.
