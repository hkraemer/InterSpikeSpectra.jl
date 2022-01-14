using InterSpikeSpectra
using RecurrenceAnalysis

using PyPlot
using Statistics
using SparseArrays
pygui(true)

using InterSpikeSpectra
using Revise
using RecurrenceAnalysis
using Test
using Random
using GLMNet
using Statistics
using SparseArrays


s = zeros(100)
period1 = 3
s[2:period1:end].= 1

spectrum, ρ = inter_spike_spectrum(s)
