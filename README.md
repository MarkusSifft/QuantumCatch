# QuantumPolyspectra: a Python Package for the Analysis and Simulation of Quantum Measurements 
by M. Sifft and D. Hägele

The QuantumPolyspectra package is open-source software for analyzing and simulating quantum measurements in terms of socalled quantum polysectra. Here we refere to the polysectra as as second to fourth order spectra (powerspectrum, bispectrum, and trispectrum). The simulation of measurement traces (integration of the stochastic master equation) is implemented via the QuTiP toolbox whereas the calculation of polyspectra from Hamiltonians or measurements traces recorded in the lab is performed as described in [this paper](https://link.aps.org/doi/10.1103/PhysRevB.98.205143) and [this paper](https://arxiv.org/abs/2011.07992) which also shows the utilization of quantum polyspectra to extract Hamiltonian paramers from a quantum measurement. 

## Documentation
The package is devided into two parts: the **generate** module and the **analyze** module. 
### Generate Module
This module connects any timeindependet stochastic master equation with its corresponding 

### Examples
Examples for every function of the package are currently added to the folder Examples

## Support
The development of the QuantumPolyspectra package is supported by the working group Spectroscopy of Condensed Matter of the Faculty of Physics and Astronomy at the Ruhr University Bochum.

## Dependencies
For the package multiple libraries are used for the numerics and displaying the results:
* NumPy
* SciPy
* Pandas
* Cachetools
* QuTiP
* MatPlotLib
* Plotly
* tqdm
* Numba
* Lmfit
* h5py
* arrayfire
* labellines
