# InterSpikeSpectra

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://hkraemer.github.io/InterSpikeSpectra.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://hkraemer.github.io/InterSpikeSpectra.jl/dev)
[![Build Status](https://github.com/hkraemer/InterSpikeSpectra.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/hkraemer/InterSpikeSpectra.jl/actions/workflows/CI.yml?query=branch%3Amain)

# InterSpikeSpectra.jl

A Julia package for obtaining inter-spike spectra from signals. It is recommended to analyze "spiky" signals with 
this method. As the authors showed in the corresponding paper (Kraemer et al. 2022, Spike spectra for recurrences) 
this method can yield reasonable results and insights into $\tau$-recurrence rate signals obtained from 
recurrence plots of the underlying signal.

# Installation
In Julia, activate your package manager via `]`+`ENTER`, then type
```julia
add https://github.com/hkraemer/InterSpikeSpectra.jl.git
```
When you now type 
```julia
status
```
you should see `InterSpikeSpectra.jl` listed. Now it is ready to use in any of your script via 
```julia
using InterSpikeSpectra

#...
```

# Functionality
The main function to call is `inter_spike_spectrum()`, simply type 
```julia
? inter_spike_spectrum
```
into you Julia-IDE and read the documentation.
