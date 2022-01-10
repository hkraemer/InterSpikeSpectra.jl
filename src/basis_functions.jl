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
    `maxλ = 15`: Determines after how many tried Lambdas the algorithm stopps.
"""
function inter_spike_spectrum(s::Vector{T}; ρ_thres::Real = 0.95, tol::Real=1e-3, maxλ::Integer=15) where {T}
    @assert 0.8 <= ρ_thres <= 1 "Optional input `ρ_thres` must be a value in the interval [0.8, 1]"
    @assert 1e-5 <= tol <= 1 "Optional input `tol` must be a value in the interval [1e-5, 1]"
    @assert 0 < maxλ <= 100 "Optional input `maxλ` must be an integer in the interval (0, 100]."

    # normalization
    s = (s.-mean(s)) ./ std(s)
    s .-= minimum(s)
    s ./= maximum(s)

    N = length(s)
    Θ = generate_basis_functions(N)'

    return compute_spectrum_according_to_threshold(s, sparse(Θ), ρ_thres, tol, maxλ)
end

"""
    Determine the right regularization parameter with respect to ρ_thres & tol
    using the Newton-method.
"""
function compute_spectrum_according_to_threshold(s::Vector, Θ::SparseMatrixCSC, ρ_thres::Real, tol::Real, maxλ::Integer)
    # initial Lambda (start with least squares solution)
    lambda_i = 0
    # initial Lambda-step
    lambda_step = 0.1

    pos = true          # indicates whether lambda_step is positive or not
    upper = false       # indicates whether an upper limit for lambda has been reached
    cond1 = false
    flag = true
    i = 0
    while flag

        lambda_f = lambda_i + lambda_step  # try new lambda
        lambda_i = lambda_f                # update lambda for the next run

        # make the regression with specific lambda
        path = glmnet(Θ, @view s[:]; lambda = [lambda_f])
        y = path.betas
        # check whether the regenerated signal matches with the given threshold
        rr = cor(vec(regenerate_signal(Θ, y)), s)

        # check whether max iterations are reached
        i += 1
        if i == maxλ + 1
            println("Algorithm stopped due to maximum number of lambdas were tried without convergence. Please increase `tol`-input OR increae `maxlambdas` and if this does not help `correlation_threshold` must be higher.")
            return zeros(length(s)), rr
            flag = false
        end
        
        # check whether upper limit is reached or convergence is fullfilled
        if isnan(rr) || (rr - ρ_thres) < 0
            upper = true
        end
        if (rr - ρ_thres) == 0 || abs(rr - ρ_thres) < tol
            spectrum = pool_frequencies(y, length(s))
            return spectrum, rr
            flag = false
        end

        # alter lamba-steps
        if !isnan(rr) && !upper
            continue
        elseif (rr - ρ_thres) < 0 || isnan(rr)
            upper = true
            if cond1
                pos = true
            else
                pos = false
            end
            cond1 = true
        elseif (rr - ρ_thres) > 0
            if cond1
                pos = false
            else
                pos = true
            end
            cond1 = false;
        elseif isnan(rr) && !upper
            upper = true
            pos = false
            cond1 = true
        end
        if pos
            lambda_step = lambda_step/2 # split the interval
        else
            lambda_step = (lambda_step*(-1))/2 # split the interval
        end
    end
end

# regenerate a decomposed signal from basis functions and its coefficients
function regenerate_signal(Θ::SparseMatrixCSC, coefs::CompressedPredictorMatrix)
    @assert size(Θ,2) == length(coefs) "Number of basis functions must match number of coefficients"
    return Θ*coefs
end

# pool the same frequencies
function pool_frequencies(y::CompressedPredictorMatrix, N::Integer)
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
