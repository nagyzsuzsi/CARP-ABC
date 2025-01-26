# Overview

This repository contains the implementation of the Artificial Bee Colony algorithm for CARP (CARP-ABC) presented in the paper __*An Artificial Bee Colony Algorithm for Static and Dynamic Capacitated Arc Routing Problems*__ by Zsuzsanna Nagy, Agnes Werner-Stark and Tibor Dulai. The paper is available [here](https://www.mdpi.com/2227-7390/10/13/2205).

It also includes a prototype implementation of the data-driven DCARP framework described in the paper __*A Data-driven Solution for The Dynamic Capacitated Arc Routing Problem*__ by Zsuzsanna Nagy, Agnes Werner-Stark and Tibor Dulai. The paper is available [here](https://www.conferences-scientific.cz/file/9788088203247) (pp. 64-83).

# Repository structure

- [inputs](inputs): Contains the CARP instance files files used for testing and experiments.
- [utils.py](utils.py): Contains the implementations of the following algorithms:
  - Minimal Rerouting (RR1) algorithm.
  - Artificial Bee Colony algorithm for CARP (CARP-ABC).
  - Hybrid Metaheuristic Approach (HMA).
  - Ant Colony Optimization algorithm with Path Relinking (ACOPR).
- [utils_mod.py](utils_mod.py): A modified version of the `utils.py` file used for the move operator experiments.
- [carp_tests.py](carp_tests.py): Contains the code that implements and runs the CARP experiments.
- [dcarp_tests.py](dcarp_tests.py): Contains the code that implements and runs the DCARP experiments.
