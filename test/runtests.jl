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

println("Begin testing InterSpikeSpectra.jl...")

@testset "Simple spike train with randomized peaks" begin
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
    spectrum, ρ = inter_spike_spectrum(test_tauRR; ρ_thres = threshold, tol = tol, regression_type=normal())
    spectrum2, ρ2 = inter_spike_spectrum(test_tauRR; ρ_thres = threshold, tol = tol)
    @test spectrum == spectrum2
    @test ρ==ρ2
    maxis, maxis_idx = get_maxima(spectrum)
    @test abs(ρ - threshold) < tol
    @test maxis_idx == [period2, period1]
    @test 0.53 < maxis[1] < 0.54
    @test 0.46 < maxis[2] < 0.47

    threshold = 0.99
    tol = 1e-3
    spectrum, ρ = inter_spike_spectrum(test_tauRR; regression_type=normal())

    maxis, max_idx = get_maxima(spectrum)
    t_idx = maxis .> 0.1
    peak_idxs = max_idx[t_idx]

    @test abs(ρ - threshold) < tol
    @test length(peak_idxs) == 2
    @test peak_idxs[1] == period2
    @test peak_idxs[2] == period1

    # Elastic net
    alpha = 0.5
    spectrum, ρ = inter_spike_spectrum(test_tauRR; alpha, regression_type=normal())
    
    maxis, max_idx = get_maxima(spectrum)
    @test max_idx == [4, 8, 13, 16, 22]
    t_idx = maxis .> 0.1
    peak_idxs = max_idx[t_idx]
    @test peak_idxs[1] == period2
    @test peak_idxs[2] == period1
    @test peak_idxs[3] == 2*period2

    # randomized peak heights
    Random.seed!(1234)
    threshold = 0.95
    tol = 1e-3
    maxcycles = 20
    test_tauRR = abs.(randn(N)) .* test_tauRR2[1,:]
    spectrum, _ = inter_spike_spectrum(test_tauRR; ρ_thres= threshold, tol = tol, max_iter = maxcycles, regression_type=normal())

    maxis, max_idx = get_maxima(spectrum)
    t_idx = maxis .> 0.23
    peak_idxs = max_idx[t_idx]

    @test sum(rem.(peak_idxs, period1)) == 0
end


@testset "LASSO & STLS" begin
    
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

    threshold = 0.99
    tol = 1e-3
    
    spectrum1, ρ1 = inter_spike_spectrum(s; ρ_thres = threshold, tol, regression_type=normal())
    spectrum2, ρ2 = inter_spike_spectrum(s; method=STLS(), ρ_thres = threshold, tol)

    maxis1, max_idx1 = get_maxima(spectrum1)
    maxis2, max_idx2 = get_maxima(spectrum2)

    @test length(maxis1) == 6
    @test length(maxis2) == 8 
    @test max_idx1 == [3,7,14,21,28,42]
    @test max_idx2 == [3,6,9,12,14,21,28,42]
    @test maxis1[1] > 0.197
    @test maxis1[4] > 0.396
    @test maxis1[6] > 0.390
    @test maxis2[1] > 0.084
    @test maxis2[6] > 0.292
    @test maxis2[8] > 0.209

end

@testset "Perfect spike train" begin

    s = zeros(100)
    period1 = 3
    s[2:period1:end].= 1

    spectrum, ρ = inter_spike_spectrum(s; regression_type=normal())

    @test ρ+ 1e-3 >= 1
    maxis, max_idx = get_maxima(spectrum)
    @test maxis[1] > 0.999
    @test length(max_idx) == 1
    @test max_idx[1] == period1

    period2 = 11
    s[5:period2:end].= .8

    tol = 1e-4
    threshold = .99
    spectrum, ρ = inter_spike_spectrum(s; tol = tol, regression_type=normal())
    @test abs(ρ - threshold) < tol
    maxis, max_idx = get_maxima(spectrum)
    @test length(max_idx) == 2
    @test max_idx[1] == period1
    @test max_idx[2] == period2 || max_idx[2] / max_idx[1] == period2
    @test maxis[1] < maxis[2]

end

@testset "Normal & logistic regression" begin

    N = 300
    M = Int.(ceil(N/2))
    period1 = 3
    period2 = 7
    s = zeros(N)
    s[period1:period1:end] .= 1
    s[period2:period2:end] .= 1
    
    threshold = 0.99
    spectrum1, rho1 = inter_spike_spectrum(s; ρ_thres = threshold, regression_type=logit())
    spectrum2, rho2 = inter_spike_spectrum(s; ρ_thres = threshold, regression_type=normal())
    
    peaks1, peaks1_idx = get_maxima(spectrum1)
    peaks2, peaks2_idx = get_maxima(spectrum2)
    
    @test abs(rho1 - threshold) < 1e-3
    @test abs(rho2 - threshold) < 1e-3
    @test 0.5633 < peaks1[1] < 0.5634
    @test 0.4366 < peaks1[2] < 0.4367
    @test 0.332 < peaks2[1] < 0.334
    @test peaks2[2] < 2e-15
    @test 0.665 < peaks2[3] < 0.667
    @test peaks1_idx == [3,7]
    @test peaks2_idx == [3,7,21]
    
    threshold = 0.9
    spectrum1, rho1 = inter_spike_spectrum(s; ρ_thres = threshold, regression_type=logit())
    spectrum2, rho2 = inter_spike_spectrum(s; ρ_thres = threshold, regression_type=normal())
    
    peaks1, peaks1_idx = get_maxima(spectrum1)
    peaks2, peaks2_idx = get_maxima(spectrum2)
    
    @test abs(rho1 - threshold) < 1e-3
    @test abs(rho2 - threshold) < 1e-3
    @test 0.5633 < peaks1[1] < 0.5634
    @test 0.4366 < peaks1[2] < 0.4367
    @test 0.5633 < peaks2[1] < 0.5634
    @test 0.4366 < peaks2[2] < 0.4367
    @test peaks1_idx == [3,7]
    @test peaks2_idx == [3,7]

end

@testset "Random input" begin
    Random.seed!(1234)
    tol = 1e-4
    maxcycles = 20
    s = randn(50)

    threshold = 0.995
    spectrum1, ρ = inter_spike_spectrum(s; ρ_thres = threshold, tol = tol, max_iter = maxcycles, regression_type=normal())
    maxis1, _ = get_maxima(spectrum1)
    numpeaks1 = length(maxis1)
    @test abs(ρ - threshold) <= tol

    threshold = 0.85
    spectrum2, ρ = inter_spike_spectrum(s; ρ_thres = threshold, regression_type=normal())
    maxis2, _ = get_maxima(spectrum2)
    numpeaks2 = length(maxis2)
    @test abs(ρ - threshold) <= 1e-3
    @test numpeaks2 == 5 && numpeaks1 == 8
    @test 0.053 < maxis1[1] < 0.056
    @test 0.089 < maxis2[1] < 0.09
end

true
