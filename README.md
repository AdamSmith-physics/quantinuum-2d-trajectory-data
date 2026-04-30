# Quantum trajectory simulation of two-dimensional non-equilibrium steady states with a trapped ion quantum processor

Data repository for the paper "Quantum trajectory simulation of two-dimensional non-equilibrium steady states with a trapped ion quantum processor". This repository contain all data files from both experimental and classical simulations and simple codes to generate, load and plot the data. Besides plotting the data from the expriment, this repository contains the cpodes to run classical simulations of Symmetric Simple Exclusion Process (SSEP), generate free-fermion trajectories and run quantum circuits on Quantinuum's "H1-1LE" backend.

## Content

The `data/` folder contains the results from Quantinuum's H1-1 simulations, noisy simulation (on Quantinuum's H1-1E and H1-Emultor devices) and classical numerics, all saved as HDF5 files. The latter include ideal simulations of trotterised cicuits and the simulations in the Lindblad limit $dt \rightarrow 0$.

The core files to run simulations are contained in `src/` and `run/`. In the former, there are all codes required to set up the free-fermion simulation and to build circuits for both hard-core bosons and fermions. The codes that effectively run the trajectories are contained in `run/`.
Note that when running circuit trajectories, the results are saved in a single HDF5 file that will be added in a `data_local/` folder, which is included in `.gitignore`.

To plot the data we use the Jupyter Notebook ``data_plotting.ipynb`` in the ``plotting/`` folder. 

## Installing packages

The ``environment.yml`` allows to set up a conda environment with the necessary packages. These include "pytket-quantinuum[pecos]", which allows access to the local noiseless emulator. To access other Quantinuum systems, remove the [pecos] extra-install-argument.

For Windows users, run this command from the Terminal to create an environment called `quantum_trajectories2d`:
```
conda env create -f environment.yml
```
On mac, the "pytket-quantinuum[pecos]" package needs to be installed separately using the line 
```
pip install "pytket-quantinuum[pecos]"
```
in the Terminal.