# Main functionality of the spike train decomposition and the obtained inter
# spike spectrum
"""
    inter_spike_spectrum(s::Vector; kwargs...) → spectrum, ρ

    Compute a spike `spectrum` from the a signal `s`, using a LASSO optimization.
    `s` will get normalized to the unity interval `s ∈ [0 1]` internally. Second
    output `ρ` is the linear Pearson correlation coefficient between the true
    input `s` and the regenerated decomposed signal.

    Keyword arguments:
    `ρ_thres = 0.95`: The agreement of the regenerated decomposed signal with the
                      true signal. This depends on the LASSO regularization parameter
                      `λ`. `λ` gets adjusted automatically with respect to `ρ_thres`.
    `tol = 1e-3`: Allowed tolerance between `ρ_thres` and `ρ`.
    `max_iter = 15`: Determines after how many tried Lambdas the algorithm stopps.
    `verbose::Bool=true`: If true, warning messages enabled.
"""
function inter_spike_spectrum(s::Vector{T}; ρ_thres::Real = 0.95, tol::Real=1e-3, max_iter::Integer=15, verbose::Bool=true) where {T}
    @assert 0.8 <= ρ_thres <= 1 "Optional input `ρ_thres` must be a value in the interval [0.8, 1]"
    @assert 1e-5 <= tol <= 1 "Optional input `tol` must be a value in the interval [1e-5, 1]"
    @assert 1 < max_iter <= 20 "Optional input `max_iter` must be an integer in the interval (1, 20]."

    # normalization
    s = (s.-mean(s)) ./ std(s)
    s .-= minimum(s)
    s ./= maximum(s)

    N = length(s)
    Θ = generate_basis_functions(N)'

    return compute_spectrum_according_to_threshold(s, sparse(Θ), ρ_thres, tol, max_iter, verbose)
end

"""
    Determine the right regularization parameter with respect to ρ_thres & tol
    using the Newton-method.
"""
function compute_spectrum_according_to_threshold(s::Vector, Θ::SparseMatrixCSC, ρ_thres::Real, tol::Real, max_iter::Integer, verbose::Bool)

    # initial Lambda-step for estimating an upper bound for λ
    lambda_step = 0.5
    lambda_max, lambda_min, y_act, ρ_act = find_lambda_max(s, Θ, lambda_step, ρ_thres)
    # bisection search
    for i = 1:max_iter
        # check whether max iterations or tolerance-level reached
        if i == max_iter
            if verbose
                println("Algorithm stopped due to maximum number of λ's were tried without convergence to the specified `ρ_thres`")
            end
            return pool_frequencies(y_act, length(s)), ρ_act
            break
        elseif abs(ρ_act - ρ_thres) <= tol
            return pool_frequencies(y_act, length(s)), ρ_act
        end

        # try new lambda
        actual_lambda = lambda_min + (lambda_max - lambda_min)/2
        # make the regression with specific lambda
        path = glmnet(Θ, @view s[:]; lambda = [actual_lambda])
        # check whether the regenerated signal matches with the given threshold
        rr = cor(vec(regenerate_signal(Θ, path.betas)), s)

        # pick the new bisection interval
        if isnan(rr)
            lambda_max = actual_lambda
        elseif rr < ρ_thres
            lambda_max = actual_lambda
            y_act[:] = path.betas
            ρ_act = rr
        elseif rr > ρ_thres
            lambda_min = actual_lambda
            ρ_act = rr
            y_act[:] = path.betas
        end
    end
end

# regenerate a decomposed signal from basis functions and its coefficients
function regenerate_signal(Θ::SparseMatrixCSC, coefs::CompressedPredictorMatrix)
    @assert size(Θ,2) == length(coefs) "Number of basis functions must match number of coefficients"
    return Θ*coefs
end

# pool the same frequencies
function pool_frequencies(y::Vector, N::Integer)
    y = abs.(y)
    spectrum = zeros(N)
    cnt = 1
    for i = 1:N
        occs = sum(y[cnt:cnt+i-1] .> 0)
        if occs>0
            spectrum[i] = sum(y[cnt:cnt+i-1]) / occs
        else
            spectrum[i] = 0
        end
        cnt += i
    end
    return spectrum
end

# estimate a maximum λ-value for which the correlation coefficient form the re-
# generated signal and the input signal falls below `ρ_thres` or is NaN.
function find_lambda_max(s::Vector, Θ::SparseMatrixCSC, lambda_step::Real, ρ_thres::Real)
    lambda = 0.
    lambda_min = 0.
    ρ_min = 1
    y_min = zeros(size(Θ,2))
    for i = 1:10000
        # make the regression with specific lambda
        path = glmnet(Θ, @view s[:]; lambda = [lambda])
        # check whether the regenerated signal matches with the given threshold
        rr = cor(vec(regenerate_signal(Θ, path.betas)), s)
        if i == 1
            y_min[:] = vec(path.betas)
            ρ_min = rr
        end

        if rr > ρ_thres
            lambda_min = lambda
            y_min[:] = vec(path.betas)
            ρ_min = rr
        elseif isnan(rr) || rr <= ρ_thres
            return lambda, lambda_min, y_min, ρ_min
            break
        end
        lambda += lambda_step
    end
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
