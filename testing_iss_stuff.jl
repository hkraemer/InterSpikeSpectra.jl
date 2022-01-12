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

@test ρ+ 1e-3 >= 1
_, max_idx = get_maxima(spectrum)
@test length(max_idx) == 1
@test max_idx[1] == period1

period2 = 11
s[5:period2:end].= .8

tol = 1e-4
threshold = .9999
spectrum, ρ = inter_spike_spectrum(s; ρ_thres=threshold, tol = tol)
@test abs(ρ - threshold) < tol
maxis, max_idx = get_maxima(spectrum)
@test length(max_idx) == 2
@test max_idx[1] == period1
@test max_idx[2] == period2
@test maxis[1] > maxis[2]

figure()
subplot(211)
plot(s)
grid()
subplot(212)
plot(spectrum)
grid()
