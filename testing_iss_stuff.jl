using InterSpikeSpectra
using RecurrenceAnalysis

using PyPlot
pygui(true)

N = 300
period1 = 12
period2 = 52
test_tauRR2 = InterSpikeSpectra.create_single_basis_function(N, period1)
test_tauRR3 = InterSpikeSpectra.create_single_basis_function(N, period2)

# all 1's
test_tauRR = abs.(ones(N).*0.7) .* test_tauRR2[11,:]
test_tauRR_ = abs.(ones(N).*0.8) .* test_tauRR3[30,:]
test_tauRR = test_tauRR .+ test_tauRR_
test_tauRR = test_tauRR ./ maximum(test_tauRR)
@time spectrum, rho = inter_spike_spectrum(test_tauRR; ρ_thres= .85)
