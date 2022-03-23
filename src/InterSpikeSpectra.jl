module InterSpikeSpectra

using Statistics, LinearAlgebra, GLMNet, SparseArrays, IterativeSolvers, StatsBase
using Revise

include("basis_functions.jl")

export inter_spike_spectrum
export STLS, lasso
export normal, logit, auto

end
