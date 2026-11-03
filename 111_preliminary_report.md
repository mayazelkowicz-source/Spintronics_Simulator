# Group 111

# Simulating an energy-efficient In-Memory Compute AI accelerator by mapping Binarized Neural Networks to Spintronic MTJ arrays

**Milestone 1 — First Report**

| | |
|---|---|
| **Group Number** | **111** |
| **Team Members** | Maya Zelkowicz (maya.zelkowicz@mail.huji.ac.il)<br>Sarit Mishiev (sarit.mishiev@mail.huji.ac.il) |
| **Mentor** | Daniella (daniellah@gmail.com) |
| **Advisor** | Dr. Amir Capua (amir.capua@mail.huji.ac.il) |
| **Organization** | HUJI Spintronics Lab, The Hebrew University of Jerusalem |
| **Git Repository** | https://github.com/mayazelkowicz-source/Spintronics_Simulator |
| **Notion Workspace** | https://app.notion.com/p/Spintronics-Lab-Simulating-an-energy-efficient-In-Memory-Compute-AI-accelerator-ce5240c4d77d83e7a51d818497730381 |

<div style="page-break-after: always;"></div>

## 1. Problem Definition

Running Artificial Intelligence (AI) on battery-powered edge devices is limited by the "Von Neumann bottleneck": moving data between memory and processor dominates energy and latency. Spintronic memories built from Magnetic Tunnel Junctions (MTJs) can compute inside memory, but thermal noise and switching errors may corrupt results. Chip architects and device physicists lack a fast way to predict this impact.

**Can AI inference stay accurate on noisy magnetic memory?**

## 2. Background and Related Work

**Background.** *Spintronics* exploits the electron's spin as well as its charge. Its key device, the **Magnetic Tunnel Junction (MTJ)**, sandwiches a thin insulator between two ferromagnetic layers: one fixed, one free to flip. Parallel (P) magnetizations give low resistance, Anti-Parallel (AP) high resistance — Tunnel Magnetoresistance [1]. The free layer obeys the **Landau–Lifshitz–Gilbert (LLG)** equation: magnetization precesses around the effective field and is damped toward it [2]. A spin-polarized current exerts **Spin-Transfer Torque (STT)**, which opposes damping and switches the bit [3]; thermal fluctuations make switching probabilistic. A **Binarized Neural Network (BNN)** restricts weights and activations to ±1 — a natural fit for two-state MTJs.

**Related work.**
- **BinaryNet / XNOR-Net** [4, 5]: ±1 networks retain much of full-precision accuracy at a fraction of the compute, assuming ideal digital hardware.
- **Samsung MRAM crossbar** [6]: a 64×64 MTJ array classified handwritten digits at 93.23% accuracy (software baseline: 95.24%). Because MTJ resistance is low, conventional *current summation* draws large currents, so the authors used *resistance summation*.
- **Hirtzlin et al.** [7]: simulated BNNs on spin-torque MRAM and showed that bit-error rates up to 0.1% barely affect accuracy, enabling cheaper cells without error correction. Device physics is abstracted into a single bit-error rate, with digital sensing.
- **NeuroSim** [8] and **IBM AIHWKit** [9]: generic analog-crossbar simulators centered on resistive and phase-change memories.

**Our position.** We extend [7] by replacing a single error rate with errors derived from LLG/STT physics and HUJI Spintronics Lab measurements: thermal noise, switching probability and resistance spread. Our behavioral crossbar model is summation-agnostic: it evaluates current summation in small tiles and complementary P/AP differential pairs, quantifying where current summation stops scaling and [6]'s resistance summation becomes necessary.

**References**

1. S. Yuasa et al., *Nature Materials* 3, 868 (2004). https://doi.org/10.1038/nmat1257
2. M. Althammer et al., "Spin Pumping," in Sato & Saitoh (eds.), *Spintronics for Next Generation Innovative Devices*, Wiley (2015). https://doi.org/10.1002/9781118751886
3. D. C. Ralph & M. D. Stiles, "Spin transfer torques," *JMMM* 320, 1190 (2008). https://doi.org/10.1016/j.jmmm.2007.12.019
4. M. Courbariaux, I. Hubara et al., "Binarized Neural Networks" (2016). https://arxiv.org/abs/1602.02830
5. M. Rastegari et al., "XNOR-Net," ECCV (2016). https://arxiv.org/abs/1603.05279
6. S. Jung et al., "A crossbar array of magnetoresistive memory devices for in-memory computing," *Nature* 601, 211 (2022). https://doi.org/10.1038/s41586-021-04196-6
7. T. Hirtzlin et al., "Implementing Binarized Neural Networks with Magnetoresistive RAM without Error Correction," IEEE/ACM NANOARCH (2019). https://arxiv.org/abs/1908.04085
8. P.-Y. Chen, X. Peng, S. Yu, "NeuroSim," *IEEE TCAD* 37(12), 3067 (2018). https://doi.org/10.1109/TCAD.2018.2789723
9. M. J. Rasch et al., "AIHWKit," IEEE AICAS (2021). https://arxiv.org/abs/2104.02184
10. M. Hu et al., "Dot-Product Engine for Neuromorphic Computing," DAC (2016). https://doi.org/10.1145/2897937.2898010
11. M. Horowitz, "Computing's energy problem," IEEE ISSCC (2014). https://doi.org/10.1109/ISSCC.2014.6757323

## 3. Proposed Solution

### Overview — alternative approaches

**Abstract view.** The core question: how much of a stored BNN's accuracy survives real device physics, and at what energy? An MTJ is a nanoscale magnetic switch whose resistance stores one bit. In a crossbar grid, inputs drive the row wires (*word lines*) and each column wire (*bit line*) sums the resulting currents — a weighted sum computed inside memory.

| | **Alt. 1: Fabricate hardware** | **Alt. 2: Circuit simulation (SPICE — Simulation Program with Integrated Circuit Emphasis)** | **Alt. 3: Hardware-aware architectural simulator (chosen)** |
|---|---|---|---|
| Pros | Ground truth | Electrically precise | Fast; thousands of noise scenarios |
| Cons | Months per iteration; costly | Hours per inference at 200,000+ devices | Needs calibration |

### Solution proposal

We choose **Alternative 3**: our question is statistical, requiring thousands of full-network inferences — feasible only architecturally. Calibration risk is mitigated by lab measurements and comparison with published circuit-level results [10].

### Key components

![Figure 1 – LLG magnetization dynamics: precession of M around H_eff with damping; (c) spin current exchanged with an adjacent layer, the reciprocal of STT.](report_figures/fig1_llg_precession_damping_saitoh.png)

*Figure 1 – Magnetization dynamics described by the LLG equation (from the course reading, Althammer et al. [2], Fig. 6.1a,c).*

Figure 1 grounds the Noise Injector: in (a) the free layer precesses and is damped into the stable P/AP states storing ±1 weights; (c) shows angular-momentum transfer, whose reciprocal, STT, writes weights. Thermal agitation of this motion causes unintended flips.

<div style="page-break-after: always;"></div>

![Figure 2 – System architecture](report_figures/fig2_system_architecture.png)

*Figure 2 – Component diagram of the proposed simulator.*

1. **BNN Training:** ±1 weights and *golden reference* data (trusted inputs, expected outputs, labels).
2. **Weight Mapper:** +1 → P, −1 → AP.
3. **MTJ Crossbar (C++ `VirtualCrossbar`):** pixels become word-line voltages; bit lines sum currents.
4. **Noise Injector:** resistance variation, thermal read noise, physics-derived bit flips.
5. **Sense & Activate:** comparator implements sign().
6. **Metrics Collector:** accuracy, Joules per inference, latency.

### Solution feasibility

- **Algorithm proven:** our PyTorch BNN (784→256→10) reaches **94.30%** on MNIST (Modified National Institute of Standards and Technology handwritten digits).
- **Core working:** C++ ideal inference reproduces the golden *logits* (raw per-class output scores) **bit-exactly** on 100 test images.
- **Fast prototyping:** Python and C++ exchange plain CSV files, so a retrained model is simulated immediately.
- **Modularity:** components share fixed interfaces, so noise models are swapped without touching the engine; ideal mode remains a permanent regression test.

### Technologies and skills

- **C++17 / CMake** (https://cmake.org) – simulation engine. *Skills:* performance-oriented C++, memory layout, Monte Carlo simulation.
- **PyTorch** (https://pytorch.org) – BNN training. *Skills:* quantization-aware training, straight-through gradient estimation.
- **Git / GitHub** – version control. *Skills:* branching, collaborative code review.
- **Notion** – project management: timeline, tasks, requirements.
- **Hardware skills:** MTJ device physics, crossbar circuits, sense amplifiers, energy estimation.

### Challenges and uncertainties

- **IR drop** (voltage loss along resistive wires) distorts bit-line currents [10], worsened by low MTJ resistance [6]. *Backup:* 64×64 tiles with first-order attenuation, or resistance summation as in [6].
- **Read disturb:** read current exerts STT and may flip cells. *Backup:* per-read flip probability.
- **Lab data availability.** *Backup:* published parameters [1, 3, 7].

### Required knowledge

- **Literature:** Cullity, *Introduction to Magnetic Materials*; Gurevich & Melkov, *Magnetization Oscillations and Waves*; Fabian, Žutić et al., *Semiconductor Spintronics*; [2, 3, 7]; Sze et al., "Efficient Processing of Deep Neural Networks" (2017).
- **Courses:** Dr. Capua's Spintronics reading course; Computer Architecture; Machine Learning; MIT 6.5930 *Hardware Architecture for Deep Learning*.

## 4. Evaluation and Verification Scheme

**Persona.** A device researcher at the HUJI Spintronics Lab asking: *"Is our measured MTJ stack good enough for AI inference, and which parameter must improve?"*

**Verification (correctness).**
- Noise disabled: logits must be **bit-exact (100%)** with the golden reference on the full **10,000-image** MNIST test set (already achieved on 100 images).
- Unit tests for each physical model against analytical limits (e.g., zero temperature ⇒ zero flips).

**Evaluation (quantitative targets).**
- **Robustness:** accuracy drop **< 5 percentage points** versus the ideal BNN under lab-calibrated noise.
- **Sensitivity sweeps:** bit-error rate 0–10%, resistance variation 0–20%; each point averaged over **≥ 30 random seeds**, reported with 95% confidence intervals.
- **Energy:** Joules per inference from cell read energy (V²·G·t) plus sensing overhead, benchmarked against a Von Neumann baseline built from published per-operation and memory-access energies [11]; target **≥ 10× lower** energy.
- **Latency:** crossbar read cycles per inference versus sequential multiply-accumulate operations.
- **Simulator speed:** full test set in **< 60 seconds** on a laptop.

**Real-world testing.** Calibrate the Noise Injector with measured lab parameters (resistance spread, damping from ferromagnetic-resonance linewidths) and test generalization on degraded "in-the-wild" inputs (noisy, rotated digits; Fashion-MNIST).

## 5. Annual Work Plan

**First steps (November–December 2026) — essential:**
- Golden-reference verification on all 10,000 test images.
- MTJ resistance mapping and bit-line current summation.
- Collect device parameters from the Spintronics Lab.

**MVP (Minimum Viable Product) — "Skateboard", Demo Day, January 2027 — essential:** our existing C++ core integrated with an MTJ conductance model and a statistical bit-flip Noise Injector, producing a live accuracy-versus-noise curve for the MNIST BNN.

**"Bike" (March–April 2027):**
- *Essential:* physics-derived noise (thermal stability, STT switching probability, read disturb); energy and latency models.
- *Nice-to-have:* tiling with IR-drop approximation.

**"Car" (May–June 2027):**
- *Essential:* lab-calibrated device-requirement map; final evaluation.
- *Nice-to-have:* Fashion-MNIST; NeuroSim comparison; noise-aware retraining.

**Responsibilities:** Maya Zelkowicz leads physical/noise modeling and machine learning; Sarit Mishiev leads C++ engine architecture and optimization; evaluation is joint.

## 6. Project Management

- **Notion** (https://app.notion.com/p/Spintronics-Lab-Simulating-an-energy-efficient-In-Memory-Compute-AI-accelerator-ce5240c4d77d83e7a51d818497730381): project timeline, task assignments per member, requirement tracking, and weekly progress updates.
- **Git** (https://github.com/mayazelkowicz-source/Spintronics_Simulator): all code is versioned on GitHub. Each member commits their own work with descriptive messages, and the README documents setup and run instructions. Before pushing, we run the C++ regression test manually to confirm bit-exact agreement with the golden reference.
