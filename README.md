# Simulating an energy-efficient In-Memory Compute AI accelerator by mapping Binarized Neural Networks to Spintronic MTJ arrays

**Group 111** — Maya Zelkowicz & Sarit Mishiev
Advisor: Dr. Amir Capua, HUJI Spintronics Lab · Mentor: Daniella

A hardware-aware simulator that runs a Binarized Neural Network (BNN, weights and activations in {+1, −1}) on virtual Magnetic Tunnel Junction (MTJ) crossbar arrays. The goal is to measure how much accuracy survives realistic MTJ physics (thermal noise, switching errors, resistance variation) and how much energy in-memory computing saves.

## Repository layout

| Path | Contents |
|---|---|
| `main.py` | PyTorch BNN training on MNIST and weight/test-vector export |
| `layer1_weights*.csv`, `layer2_weights*.csv` | Exported binary weight matrices |
| `test_inputs_pm1.csv`, `test_expected_logits.csv`, `test_labels.csv` | Golden reference test vectors (first 100 MNIST test images) |
| `simulator_core/` | C++17 simulator: `VirtualCrossbar`, ideal forward pass, regression test |
| `PROGRESS_REPORT.md` | Milestone progress report |
| `111_preliminary_report.md` / `.pdf` | Milestone 1 report |

## 1. Train the BNN and export weights (`main.py`)

### Requirements

- Python 3.9+
- PyTorch, torchvision, NumPy

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy
```

### Run

```bash
python3 main.py
```

What it does:

1. **Downloads MNIST** into `./data/` (about 60 MB; ignored by Git).
2. **Trains** a 784 → 256 → 10 binarized multilayer perceptron for 20 epochs (batch 128, Adam, cosine learning-rate schedule, seed 0). Weights, inputs and hidden activations are strictly ±1; the non-differentiable sign() is trained with a Straight-Through Estimator. There is no bias and no BatchNorm, so each layer maps 1:1 onto a crossbar. It uses a GPU if one is available, otherwise the CPU.
3. **Exports the binarized weights** in crossbar layout (rows = inputs/word lines, columns = outputs/bit lines):
   - `layerN_weights.csv` — values in {+1, −1}
   - `layerN_weights_bin.csv` — 0/1 encoding (+1 → 1, −1 → 0), e.g. for P/AP state mapping
   - Shapes: layer 1 is 784 × 256, layer 2 is 256 × 10.
4. **Checks the CSVs** with an integer-only NumPy "golden model" (`x @ W1 → sign → @ W2 → argmax`) on the full test set and prints its accuracy next to the PyTorch accuracy.
5. **Writes golden test vectors** for the C++ simulator: the first 100 binarized test images, their expected integer logits and their labels.

> ⚠️ Running `main.py` **overwrites** the committed CSV files in the repository root. Results can differ slightly between machines (for example GPU vs. CPU). The test vectors are regenerated together with the weights, so the C++ regression test stays consistent.

## 2. Build and run the C++ simulator

Requirements: a C++17 compiler (g++ or clang++) and CMake ≥ 3.10.

```bash
cmake -S simulator_core -B simulator_core/build
cmake --build simulator_core/build
./simulator_core/build/bnn_sim          # run from the repository root
```

The simulator loads both weight matrices into `VirtualCrossbar` instances and runs ideal (noise-free) inference on the golden test vectors. It compares the logits with `test_expected_logits.csv`. It exits with code `0` only if all logits match exactly, so it doubles as a regression test. Optional arguments: `bnn_sim <layer1.csv> <layer2.csv>`.

Expected output (abridged):

```
Ideal inference on 100 test samples:
  Logits matching golden reference: 100 / 100
  Classification accuracy:          96 / 100 (96%)
[PASS] Simulator is bit-exact with the golden reference.
```

## Status

- ✅ BNN trained (94.30% MNIST test accuracy) and weights exported
- ✅ C++ crossbar core with ideal inference, bit-exact with the golden reference
- ⬜ MTJ resistance/current model and noise injection (next milestone)

## Contributing

Each team member commits their own work with descriptive messages. Before pushing, run `bnn_sim` and confirm the `[PASS]` line. There is no automated CI; this check is manual.
