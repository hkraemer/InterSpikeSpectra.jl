using InterSpikeSpectra
using RecurrenceAnalysis

using PyPlot
pygui(true)

N = 50
period1 = 13
period2 = 8
test_tauRR2 = InterSpikeSpectra.create_single_basis_function(N, period1)
test_tauRR3 = InterSpikeSpectra.create_single_basis_function(N, period2)

# all 1's
test_tauRR = abs.(ones(N).*0.7) .* test_tauRR2[3,:]
test_tauRR_ = abs.(ones(N).*0.8) .* test_tauRR3[7,:]
test_tauRR = test_tauRR .+ test_tauRR_
test_tauRR = test_tauRR ./ maximum(test_tauRR)

spectrum, ρ = InterSpikeSpectra.inter_spike_spectrum(test_tauRR; ρ_thres = 0.85)
@test 0.83 <= ρ < 0.87

spectrum, ρ = InterSpikeSpectra.inter_spike_spectrum(test_tauRR)

maxis, max_idx = get_maxima(spectrum)
t_idx = maxis .> 0.4
peak_idxs = max_idx[t_idx]

@test 0.989 <= ρ < 0.991
@test length(peak_idxs) == 2
@test peak_idxs[1] == period2
@test peak_idxs[2] == period1

# randomized peak heights
test_tauRR = abs.(randn(N)) .* test_tauRR2[1,:]
test_tauRR /= maximum(test_tauRR)
@time spectrum, _ = InterSpikeSpectra.inter_spike_spectrum(test_tauRR)

figure()
plot(test_tauRR)

s = test_tauRR
N = length(s)
Θ = InterSpikeSpectra.generate_basis_functions(N)'
Lf = fit(LassoPath, Θ, s; standardize = false, intercept = false, λ=[0.001])
ys = Lf.coefs

y, ρ = pick_the_right_coefs(ys, s, sparse(Θ), ρ_thres)





maxis, max_idx = get_maxima(spectrum)
t_idx = maxis .> 0.01
peak_idxs = max_idx[t_idx]

@test sum(rem.(peak_idxs, period1)) == 0
