# Measurement-device agnostic quantum tomography - data and code

* Data and simulations for the "Measurement-device agnostic quantum tomography" paper.
* This repository contains simulations, acquisition software and data processing scripts.
---


### Repository structure

* **`experiment/`**
  Contains data, notebooks, and analysis scripts from real experiments.

  * **`2023_08_31_selfcalib_final/`** - single-qubit experiment, including measurement data and analysis notebooks.
  * **`two_qubit_experiment/`** - two-qubit experiment.

* **`simulations/`**
  Simulated use of the self-calibration method in various settings.
  
  * **`additive_errors/`** - Simulations testing robustness of tomography under additive measurement noise.
  * **`noise-analysis/`** - Tools and scripts for analyzing the effect of different noise sources on self-calibration performance.
  * **`path_qubit/`** - Simulations of a single path-encoded qubit and the self-calibration of phase shifters.
  * **`rotating_waveplate/`** - Models and analysis for polarization tomography using a single rotating quarter-wave plate.
  * **`scaling_sim_statetomo/`** - Investagation of how artifact severity scales with the number of qubits and how we can compensate for it.
