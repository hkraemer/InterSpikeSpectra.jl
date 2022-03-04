using Pkg
Pkg.activate(".")
using InterSpikeSpectra
using RecurrenceAnalysis
using SparseArrays
using DelimitedFiles
using PyPlot
using Statistics
using BenchmarkTools
using Random
using LinearAlgebra
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
@btime spectrum, ρ = inter_spike_spectrum(s; ρ_thres = threshold, tol = tol)
@btime spectrum2, ρ2 = InterSpikeSpectra.inter_spike_spectrum2(s; ρ_thres = threshold, tol = tol)
@btime spectrum3, ρ3 = InterSpikeSpectra.inter_spike_spectrum3(s; ρ_thres = threshold, tol = tol)

using InterSpikeSpectra
spectrum, ρ = inter_spike_spectrum(s; ρ_thres = threshold, tol = tol)
# spectrum2, ρ2 = InterSpikeSpectra.inter_spike_spectrum2(s; ρ_thres = threshold, tol = tol)
# spectrum3, ρ3 = InterSpikeSpectra.inter_spike_spectrum3(s; ρ_thres = threshold, tol = tol)
begin
    figure()
    subplot(211)
    plot(1:N,s)
    grid()
    subplot(212)
    plot(1:M,spectrum)
    grid()
end

## 
using InterSpikeSpectra
threshold = 0.99
tol = 1e-3


N = 100
s = zeros(N)
period2 = 7
s[1:period2:end] .=1

@btime spectrum1, ρ = inter_spike_spectrum(s; ρ_thres = threshold, tol)
@btime spectrum2, ρ2 = inter_spike_spectrum(s; method="STLS", ρ_thres = threshold, tol)

@time spectrum1, ρ = inter_spike_spectrum(s; ρ_thres = threshold, tol)
@time spectrum2, ρ2 = inter_spike_spectrum(s; method="STLS", ρ_thres = threshold, tol)

maxis1, max_idx1 = get_maxima(spectrum1)
maxis2, max_idx2 = get_maxima(spectrum2)

@test length(maxis1) == 6
@test length(maxis2) == 8 
@test max_idx1 == [3,7,14,21,28,42]
@test max_idx2 == [3,6,9,12,14,21,28,42]
@test maxis1[1] > 0.324
@test maxis1[4] > 0.326
@test maxis1[6] > 0.322
@test maxis2[1] > 0.161
@test maxis2[6] > 0.279
@test maxis2[8] > 0.100



N = length(s)
M = Int(ceil(N/2))
begin
    figure()
    subplot(221)
    plot(1:N,s)
    grid()
    subplot(222)
    plot(1:N,s)
    grid()
    subplot(223)
    plot(1:M,spectrum1)
    grid()
    subplot(224)
    plot(1:M,spectrum2)
    grid()
end


using LinearAlgebra
using PyPlot
pygui(true)

N = 100
s = zeros(N)
period2 = 7
s[1:period2:end] .=1
s = (s.-mean(s)) ./ std(s)
s .-= minimum(s)
s ./= maximum(s)

N = length(s)
Θ = InterSpikeSpectra.generate_basis_functions(N)'

Θ = sparse(Θ)
A = Θ
y1 = qr(Matrix(A), Val(true))\s
y2 = qr(A; ordering = Int32(9))\s
y3 = pinv(Matrix(A))*s

N = length(s)
M = Int(N/2)
yy1 = InterSpikeSpectra.pool_frequencies(y1, N)
yy2 = InterSpikeSpectra.pool_frequencies(y2, N)
yy3 = InterSpikeSpectra.pool_frequencies(y3, N)

begin
    y2 = qr(A; ordering = Int32(1))\s
    yy2 = InterSpikeSpectra.pool_frequencies(y2, N)
    figure()
    subplot(411)
    plot(1:N,s)
    grid()
    subplot(412)
    plot(1:M,yy1)
    grid()
    subplot(413)
    plot(1:M,yy2)
    grid()
    subplot(414)
    plot(1:M,yy3)
    grid()
end


