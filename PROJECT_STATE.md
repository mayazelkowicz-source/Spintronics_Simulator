# PROJECT_STATE — Spintronic BNN Inference Simulator

> ## ⚠️ Standing instructions for the AI assistant (Claude)
> 1. **Continuously append** an entry to the *Milestone Log* below whenever a major
>    milestone of this project is completed (new module, new physics model, validation
>    result, architectural change, etc.). Never rewrite or delete past entries; add new ones
>    at the bottom with the date, what changed, and the current status.
> 2. **Scope rule — strict:** this file is **EXCLUSIVELY** for the Spintronics / BNN-to-MTJ
>    simulator project. Do **not** log, mention, or document any unrelated queries, side
>    tasks, or general questions asked in this directory.
> 3. Keep the *Current Architecture* and *Roadmap* sections in sync with reality.

---

## 1. Overarching Goal
A year-long B.Sc. final project: an **architectural C++ inference simulator for a
spintronic AI accelerator**. A Binarized Neural Network (BNN) trained offline is mapped
onto virtual crossbar arrays of **Magnetic Tunnel Junction (MTJ)** devices, where each
binary weight (+1 / −1) corresponds to an MTJ state (e.g. parallel / anti-parallel).
The simulator will execute in-memory MAC inference and, in later stages, model MTJ
device physics, variability, noise and error injection to study their effect on accuracy.

## 2. Initial State (2026-10-20)
- **Training side (Python / PyTorch)** — `main.py` (project root):
  binarized MLP on MNIST, topology **784 → 256 → 10**, no bias, sign activations
  (sign(0) → +1), weights/inputs/hidden activations strictly in {+1, −1}.
- **Exported artifacts (crossbar layout: rows = inputs, cols = outputs):**
  - `layer1_weights_bin.csv` — 784 × 256, 0/1 encoding (1 → +1, 0 → −1)
  - `layer2_weights_bin.csv` — 256 × 10, 0/1 encoding
  - `layer1_weights.csv`, `layer2_weights.csv` — same matrices in ±1 encoding
  - `test_inputs_pm1.csv` (100 × 784 ±1 inputs), `test_expected_logits.csv`,
    `test_labels.csv` — golden reference for validating the C++ simulator.

## 3. Current Architecture
```
simulator_core/
├── CMakeLists.txt                # C++17 build, target: bnn_sim
├── VirtualCrossbar.h/.cpp        # 2D array of binary cells (std::vector<int8_t>, row-major)
├── VirtualCrossbarInference.cpp  # Ideal forward pass + sign activation
├── CsvUtils.h/.cpp               # Generic integer-CSV reader (test vectors)
└── main.cpp                      # Loads layers, runs inference, validates vs. golden data
```
- **`VirtualCrossbar`** — named `rows × cols` array of cells, each storing +1 or −1.
  `loadFromCsv()` parses 0/1 or ±1 CSVs, validates the shape strictly, and only commits
  on a fully valid file. Accessors: `at()`, `set()`, `rows()`, `cols()`, `printSample()`.
- **`VirtualCrossbar::forward(x)`** — ideal, noise-free bit-line column sums
  `y[c] = Σ_r x[r]·W[r][c]` (`y = x @ W`) for x ∈ {+1, −1}^rows; returns `int32` sums.
- **`signActivation(y)`** — inter-layer comparator, +1 if y ≥ 0 else −1 (sign(0) → +1).
- **Inference pipeline:** `layer2.forward(signActivation(layer1.forward(x)))` → 10 raw
  logits → argmax.
- No MTJ physics, noise, or error injection yet (intentional for this stage).

**Build & run** (from project root):
```bash
cmake -S simulator_core -B simulator_core/build && cmake --build simulator_core/build
./simulator_core/build/bnn_sim                 # optional args: <layer1.csv> <layer2.csv>
# Exit code 0 = logits bit-exact with test_expected_logits.csv
```

## 4. Roadmap (tentative)
1. ✅ Core infrastructure & weight storage in virtual crossbars
2. ✅ Ideal digital inference (MAC + sign) — bit-exact with `test_expected_logits.csv`
3. ⬜ MTJ device model (R_P / R_AP, TMR, read currents, column sensing)
4. ⬜ Non-idealities: device variability, read/write errors, noise injection
5. ⬜ Accuracy-vs-non-ideality experiments and architectural metrics (energy, latency, area)

---

## 5. Milestone Log
### 2026-10-20 — Milestone 1: Simulator core skeleton & weight loading
- Created `simulator_core/` with `VirtualCrossbar` class (`.h`/`.cpp`), `main.cpp`, CMake build.
- Both layers load successfully (784×256 and 256×10); 2×2 samples verified against the
  ±1 reference CSVs (`layer*_weights.csv`).
- Status: ✅ complete. Next: ideal inference pass validated against golden logits.

### 2026-10-20 — Milestone 2: Ideal (noise-free) inference
- Milestone 1 (core infrastructure & weight loading) confirmed complete; removed the
  duplicate `PythonProject/` directory — `main.py` at the root is the single source.
- Added `VirtualCrossbar::forward()` (ideal matrix-vector product over ±1 weights) and
  `signActivation()` in `VirtualCrossbarInference.cpp`, plus `CsvUtils` for test vectors.
- `main.cpp` runs the full 784 → 256 → 10 network on the 100 golden test samples:
  **100/100 logit vectors bit-exact** with `test_expected_logits.csv`,
  **96/100 (96%) classification accuracy** against `test_labels.csv`.
- Status: ✅ complete. Next: MTJ device model (R_P / R_AP mapping, column sensing).
