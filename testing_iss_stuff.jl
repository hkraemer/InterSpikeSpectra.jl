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
using BenchmarkTools
using SparseArrays

using DynamicalSystemsBase
using InterSpikeSpectra
using RecurrenceAnalysis
using DelimitedFiles
using DelayEmbeddings
using Random

using PyPlot
pygui(true)

## We illustrate the Inter Spike Spectrum for the paradigmatic Roessler system


ds = Systems.lorenz()
data = trajectory(ds,200)
data1 = data[15001:end,:]

figure()
plot(data1[:,1])

RP1 = RecurrenceAnalysis.RecurrenceMatrix(data1[1:5000,:], ε; fixedrate = true)
τ_rr1 = RecurrenceAnalysis.tau_recurrence(RP1)
τ_rr1 = τ_rr1[1:N] ./ maximum(τ_rr1[1:N])

figure()
plot(τ_rr1)

threshold = 0.85
tol = 1e-3

M = 200
@time spectrum, ρ = inter_spike_spectrum(τ_rr1[1:M]; ρ_thres = threshold, tol = tol)
@time spectrum2, ρ2 = InterSpikeSpectra.inter_spike_spectrum2(τ_rr1[1:M]; ρ_thres = threshold, tol = tol)

using BenchmarkTools
