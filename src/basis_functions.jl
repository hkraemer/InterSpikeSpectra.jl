# Main functionality of the spike train decomposition and the obtained inter
# spike spectrum
"""
    inter_spike_spectrum(s::Vector) → spectrum, ρ

    Compute a spike `spectrum` from the a signal `s`, using a LASSO optimization.
    `s` will get normalized to the unity interval `s ∈ [0 1]` internally. Second
    output `ρ` is the linear Pearson correlation coeffiecient between the true
    input `s` and the regenerated decomposed signal.

    Keyword arguments:
    `ρ_thres=0.95`: The agreement of the regenerated decomposed signal with the
                    true signal. This depends on the LASSO regularization parameter
                    `λ`. `λ` gets adjusted automatically with respect to `ρ_thres`.
"""
function inter_spike_spectrum(s::Vector{T}; ρ_thres::Real = 0.95) where {T}
    @assert 0 < ρ_thres <= 1

    # normalization
    s .-= minimum(s)
    s ./= maximum(s)

    N = length(s)
    Θ = generate_basis_functions(N)'
    path = glmnet(Θ, s)
    # fit(LassoPath, Θ, s; standardize = false, intercept = false)
    ys = path.betas

    y, ρ = pick_the_right_coefs(ys, s, sparse(Θ), ρ_thres)

    # pool the same frequencies
    spectrum = zeros(N)
    cnt = 1
    for i = 1:N
        occs = sum(x->x>0, y[cnt:cnt+i-1])
        if occs>0
            spectrum[i] = sum(y[cnt:cnt+i-1])/occs
        else
            spectrum[i] = 0
        end

        cnt += i
    end
    return spectrum, ρ
end


"""
    generate_basis_functions(N::Int) → basis

    Generate a matrix `basis` with all possible `sum(1:N)` spike-basis functions.
    These functions are not independent, since we account for time shifts. Thus,
    `basis` is a set of linear-dependent basis- functions.
"""
function generate_basis_functions(N::Int)
    @assert N > 0
    num_of_basis_functions = sum(1:N)
    basis = zeros(num_of_basis_functions, N)
    cnt = 1
    for i = 1:N
       basis[cnt:cnt+i-1,:] = create_single_basis_function(N, i);
       cnt = cnt + i;
    end
    return basis
end

"""
    Create a spike-basis functions of length `N` and inter-spike-interval
    `period`. Due to time shifts, there are `period` different, linearly dependent
    basis functions.
"""
function create_single_basis_function(N::Int, period::Int)
    @assert N ≥ period
    @assert N > 0
    @assert period > 0

    cs = zeros(period, N)
    for i = 1:period:N
        cs[1,i] = 1
    end

    css = vcat(cs[1,:], zeros(period-1))
    for j = 1:period-1
        inp_array = circshift(css,j)
        cs[j+1,:] = inp_array[1:N]
    end
    return cs
end

# compute those coeffs, which would return in a regenerated signal, from which
# the correlation to the true signal is closest to ρ_thres
function pick_the_right_coefs(ys::CompressedPredictorMatrix, s::Vector, Θ::SparseMatrixCSC, ρ_thres::Real)
    N, M = size(ys)
    @assert size(Θ,2) == N
    @assert size(Θ,1) == length(s)

    ρs = zeros(M-1)
    for i = 2:M
        ρs[i-1] = cor(regenerate_signal(Θ, view(ys, ((i-1)*N)+1:i*N)), s)
    end
    d = abs.(ρ_thres .- ρs)
    min_idx = argmin(d)
    return ys[:,min_idx+1], ρs[min_idx]
end

# regenerate a decomposed signal from basis functions and its coefficients
function regenerate_signal(Θ::Union{SparseMatrixCSC, Matrix}, coefs::Union{SparseVector, Vector, SubArray})
    @assert size(Θ,2) == length(coefs)
    return Θ*coefs
end
