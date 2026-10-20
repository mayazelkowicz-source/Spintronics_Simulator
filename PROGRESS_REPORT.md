# Spintronics AI Accelerator Simulator - Progress Report

This document tracks the development milestones of the Binarized Neural Network (BNN) to MTJ hardware-aware inference simulator.

## Milestone 0: Foundational Research & Spintronics Theory

**Status:** Completed

* Completed a comprehensive theoretical study of the Spintronics course materials provided by Prof. Amir Capua.


* Researched the fundamental physics of magnetic memory devices, including spin dynamics, exchange interaction, magnetic domains, and single-domain particle behavior.


* Analyzed the core technologies of the Spintronics Lab, focusing specifically on Magnetic Tunnel Junctions (MTJ) and STT-MRAM for In-Memory Computing and neuromorphic AI applications.



## Milestone 1: BNN Training & Weights Extraction (Software Phase)

**Status:** Completed

* Trained a Binarized Neural Network (BNN) in PyTorch for image classification (MNIST dataset).
* Extracted discrete binary weights (+1 / -1) to simulate the parallel and anti-parallel resistance states of MTJ devices.
* Generated "Golden Test Vectors" (test inputs, expected logits, and true labels) to serve as a baseline for verifying the C++ hardware simulator's accuracy.

## Milestone 2: C++ Simulator Core & Ideal Inference

**Status:** Completed

* **Virtual Crossbar Infrastructure:** Developed the `VirtualCrossbar` C++ class, creating an efficient 2D array data structure to simulate the memory device layout.
* **Data Mapping:** Implemented file-parsing mechanisms to load and map the extracted binary CSV weights directly into the virtual MTJ crossbar matrix.
* **Ideal Forward Pass:** Implemented matrix-vector multiplication (dot product) and sign activation functions for multi-layer inference.
* **Verification:** Built a regression testing suite (`main.cpp`) that processes the 100 golden test vectors through the C++ simulator. The simulator achieved a 100% computational match with the golden reference model and successfully classified 96/100 images under ideal, noise-free conditions.

## Next Steps (Milestone 3)

* Integrate physical MTJ constraints, including thermal noise and switching inaccuracies (Bit-flips) into the simulation loop.
* Benchmark inference accuracy degradation against varying levels of injected physical noise.
