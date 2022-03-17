module InterSpikeSpectra

using Statistics, LinearAlgebra, GLMNet, SparseArrays, IterativeSolvers
using Revise
include("basis_functions.jl")

export inter_spike_spectrum

end
