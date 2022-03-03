using Pkg
Pkg.activate(".")
using InterSpikeSpectra
using RecurrenceAnalysis
using DelimitedFiles
using PyPlot
using Statistics
using BenchmarkTools
using Random
pygui(true)



# using SparseArrays
# pygui(true)

# using InterSpikeSpectra
# using Revise
# using RecurrenceAnalysis
# using Test
# using Random
# using GLMNet
# using Statistics
# using SparseArrays

# using SparseArrays

# using DynamicalSystemsBase
# using InterSpikeSpectra
# using RecurrenceAnalysis

# using DelayEmbeddings
# using Random

# using PyPlot
# pygui(true)

## We illustrate the Inter Spike Spectrum for the paradigmatic Roessler system

Random.seed!(1234)

N = 300
s = zeros(N)
period1 = 3
period2 = 7
period3 = 14
period4 = 87
s[2:period1:end].= 1
s[1:period2:end] .=1
s[9:period3:end] .= 1
s[8:period4:end] .= 1

s += 0.05.*randn(N)

#s = readdlm("test_sig.csv")
#s = s[1:300]



N = length(s)
M = Int(ceil(N/2))

threshold = 0.99
tol = 1e-3
@btime spectrum, ρ = inter_spike_spectrum(s; ρ_thres = threshold, tol = tol)
@btime spectrum2, ρ2 = InterSpikeSpectra.inter_spike_spectrum2(s; ρ_thres = threshold, tol = tol)
@btime spectrum3, ρ3 = InterSpikeSpectra.inter_spike_spectrum3(s; ρ_thres = threshold, tol = tol)

using InterSpikeSpectra
spectrum, ρ = inter_spike_spectrum(s; ρ_thres = threshold, tol = tol)
spectrum2, ρ2 = InterSpikeSpectra.inter_spike_spectrum2(s; ρ_thres = threshold, tol = tol)
spectrum3, ρ3 = InterSpikeSpectra.inter_spike_spectrum3(s; ρ_thres = threshold, tol = tol)

figure()
subplot(231)
plot(1:N,s)
grid()
subplot(232)
plot(1:N,s)
grid()
subplot(233)
plot(1:N,s)
grid()
subplot(234)
plot(1:M,spectrum[1:M])
grid()
subplot(235)
plot(1:M,spectrum2[1:M])
grid()
subplot(236)
plot(1:M,spectrum3[1:M])
grid()


##
using RecurrenceAnalysis
using InterSpikeSpectra
ds = Systems.lorenz(ρ=25)
data = trajectory(ds,200)
data1 = data[15001:end,:]

figure()
plot(data1[:,1])

N=200
ε = 0.05
RP1 = RecurrenceAnalysis.RecurrenceMatrix(data1[1:5000,:], ε; fixedrate = true)
τ_rr1 = RecurrenceAnalysis.tau_recurrence(RP1)
#τ_rr1 = τ_rr1[1:N] ./ maximum(τ_rr1[1:N])

figure()
plot(τ_rr1)

threshold = 0.99
tol = 1e-3

M = 200

@time spectrum, ρ = inter_spike_spectrum(τ_rr1[1:M]; ρ_thres = threshold, tol = tol)
@time spectrum2, ρ2 = InterSpikeSpectra.inter_spike_spectrum2(τ_rr1[1:M]; ρ_thres = threshold, tol = tol)
@time spectrum3, ρ3 = InterSpikeSpectra.inter_spike_spectrum3(τ_rr1[1:M]; ρ_thres = threshold, tol = tol)

MM = Int(ceil(M/2))

spectrum == spectrum2
spectrum3 == spectrum2[1:MM]
spectrum3 == spectrum[1:MM]

s = τ_rr1[1:M]
figure()
subplot(231)
plot(1:M,s)
grid()
subplot(232)
plot(1:M,s)
grid()
subplot(233)
plot(1:M,s)
grid()
subplot(234)
plot(1:MM,spectrum[1:MM])
grid()
subplot(235)
plot(1:M,spectrum2[1:M])
grid()
subplot(236)
plot(1:M,spectrum3[1:M])
grid()

##
using Statistics, LinearAlgebra, GLMNet, SparseArrays, Revise

N = 100
s = zeros(N)
period1 = 3
period2 = 7
period3 = 14
period4 = 87
s[2:period1:end].= 1
s[1:period2:end] .=1
s[9:period3:end] .= 1
s[8:period4:end] .= 1

s = (s.-mean(s)) ./ std(s)
s .-= minimum(s)
s ./= maximum(s)

N = length(s)
truncate_idx = sum(1:Int(ceil(N/2)))
Θ = InterSpikeSpectra.generate_basis_functions(N)[1:truncate_idx,:]'

##

using InterSpikeSpectra
using RecurrenceAnalysis
using Test
using Random

Random.seed!(1234)
N = 50
period1 = 13
period2 = 8
test_tauRR2 = InterSpikeSpectra.create_single_basis_function(N, period1)
test_tauRR3 = InterSpikeSpectra.create_single_basis_function(N, period2)

# all 1's
test_tauRR = abs.(ones(N).*0.7) .* test_tauRR2[3,:]
test_tauRR_ = abs.(ones(N).*0.8) .* test_tauRR3[7,:]
test_tauRR = test_tauRR .+ test_tauRR_
test_tauRR = test_tauRR + 0.05.*randn(N)

threshold = 0.85
tol = 1e-3
spectrum, ρ = inter_spike_spectrum(test_tauRR; ρ_thres = threshold, tol = tol)
@test abs(ρ - threshold) < tol

threshold = 0.99
tol = 1e-3
spectrum, ρ = inter_spike_spectrum(test_tauRR)

maxis, max_idx = get_maxima(spectrum)
t_idx = maxis .> 0.1
peak_idxs = max_idx[t_idx]

@test abs(ρ - threshold) < tol
@test length(peak_idxs) == 2
@test peak_idxs[1] == period2
@test peak_idxs[2] == period1

# randomized peak heights
Random.seed!(1234)
threshold = 0.95
tol = 1e-3
maxcycles = 20
test_tauRR = abs.(randn(N)) .* test_tauRR2[1,:]
spectrum, _ = inter_spike_spectrum(test_tauRR; ρ_thres= threshold, tol = tol, max_iter = maxcycles)

maxis, max_idx = get_maxima(spectrum)
t_idx = maxis .> 0.2
peak_idxs = max_idx[t_idx]

@test sum(rem.(peak_idxs, period1)) == 0

figure()
subplot(211)
plot(1:length(test_tauRR), test_tauRR)
grid()
subplot(212)
plot(1:length(spectrum), spectrum)
grid()
