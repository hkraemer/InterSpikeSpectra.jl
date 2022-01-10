using InterSpikeSpectra
using RecurrenceAnalysis

using PyPlot
using Statistics
using SparseArrays
pygui(true)

using InterSpikeSpectra
using RecurrenceAnalysis
using Test
using Random
using GLMNet


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


##
s = vec(readdlm("test_sig.csv"))


s = (s.-mean(s)) ./ std(s)
s .-= minimum(s)
s ./= maximum(s)
xs = deepcopy(s)

N = length(s)
Θ = InterSpikeSpectra.generate_basis_functions(N)'

lambda_f = 0.1
path = glmnet(sparse(Θ), view xs[:]; lambda = [lambda_f])
y = path.betas
# check whether the regenerated signal matches with the given

rrr = cor(vec(InterSpikeSpectra.regenerate_signal(sparse(Θ), y)), s)


threshold = 0.85
tol = 1e-2
spectrum, ρ = inter_spike_spectrum(test_tauRR; ρ_thres = threshold, tol = tol, maxλ=10)
@test 0.83 <= ρ < 0.88


figure()
plot(s)
grid()


spectrum, ρ = inter_spike_spectrum(test_tauRR; ρ_thres = 0.99)

maxis, max_idx = get_maxima(spectrum)
t_idx = maxis .> 0.3
peak_idxs = max_idx[t_idx]

@test 0.989 <= ρ < 0.991
@test length(peak_idxs) == 2
@test peak_idxs[1] == period2
@test peak_idxs[2] == period1
