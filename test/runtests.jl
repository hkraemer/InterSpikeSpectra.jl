using InterSpikeSpectra
using RecurrenceAnalysis
using Test
using Random

"""
Return the maxima of the given time series s and its indices
"""
function get_maxima(s::Vector{T}) where {T}
    maximas = T[]
    maximas_idx = Int[]
    N = length(s)
    flag = false
    first_point = 0
    for i = 2:N-1
        if s[i-1] < s[i] && s[i+1] < s[i]
            flag = false
            push!(maximas, s[i])
            push!(maximas_idx, i)
        end
        # handling constant values
        if flag
            if s[i+1] < s[first_point]
                flag = false
                push!(maximas, s[first_point])
                push!(maximas_idx, first_point)
            elseif s[i+1] > s[first_point]
                flag = false
            end
        end
        if s[i-1] < s[i] && s[i+1] == s[i]
            flag = true
            first_point = i
        end
    end
    # make sure there is no empty vector returned
    if isempty(maximas)
        maximas, maximas_idx = findmax(s)
    end
    return maximas, maximas_idx
end

@testset "InterSpikeSpectra.jl" begin
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
    @test 0.83 <= ρ < 0.88

    spectrum, ρ = InterSpikeSpectra.inter_spike_spectrum(test_tauRR)

    maxis, max_idx = get_maxima(spectrum)
    t_idx = maxis .> 0.3
    peak_idxs = max_idx[t_idx]

    @test 0.989 <= ρ < 0.991
    @test length(peak_idxs) == 2
    @test peak_idxs[1] == period2
    @test peak_idxs[2] == period1

    # randomized peak heights
    Random.seed!(1234)
    test_tauRR = abs.(randn(N)) .* test_tauRR2[1,:]
    test_tauRR /= maximum(test_tauRR)
    @time spectrum, _ = InterSpikeSpectra.inter_spike_spectrum(test_tauRR; ρ_thres=0.95)

    maxis, max_idx = get_maxima(spectrum)
    t_idx = maxis .> 0.01
    peak_idxs = max_idx[t_idx]

    @test sum(rem.(peak_idxs, period1)) == 0
end
