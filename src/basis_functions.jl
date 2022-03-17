# Main functionality of the spike train decomposition and the obtained inter
# spike spectrum

# Define the two sparse regression techniques to be used and two regression types
abstract type AbstractRegressionMethod end
abstract type AbstractRegressionType end

struct lasso <: AbstractRegressionMethod end
struct STLS <: AbstractRegressionMethod end

struct logit <: AbstractRegressionType end

"""
    inter_spike_spectrum(s::Vector; kwargs...) → spectrum, ρ

    Compute a spike `spectrum` from the a signal `s`, using a LASSO optimization.
    `s` will get normalized to the unity interval `s ∈ [0 1]` internally. Second
    output `ρ` is the linear Pearson correlation coefficient between the true
    input `s` and the regenerated decomposed signal.

    Keyword arguments:
    `method::String = "lasso"` : The method for sparse regression. Pick either "lasso" or 
                                 "STLS" (sequential thresholded least squares)
    `ρ_thres = 0.99`: The agreement of the regenerated decomposed signal with the
                      true signal. This depends on the LASSO regularization parameter
                      `λ`. `λ` gets adjusted automatically with respect to `ρ_thres`.
    `tol = 1e-3`: Allowed tolerance between `ρ_thres` and `ρ`.
    `max_iter = 15`: Determines after how many tried Lambdas the algorithm stopps.
    `verbose::Bool=true`: If true, warning messages enabled.
"""
function inter_spike_spectrum(s::Vector{T}; method::String="lasso", ρ_thres::Real = 0.99, tol::Real=1e-3, max_iter::Integer=15, verbose::Bool=true) where {T}
    @assert 0.8 <= ρ_thres <= 1 "Optional input `ρ_thres` must be a value in the interval [0.8, 1]"
    @assert 1e-5 <= tol <= 1 "Optional input `tol` must be a value in the interval [1e-5, 1]"
    @assert 1 < max_iter <= 20 "Optional input `max_iter` must be an integer in the interval (1, 20]."
    @assert (method == "lasso" || method == "STLS") "A valid regression method must be chosen (lasso or STLS)"

    # normalization
    s = (s.-mean(s)) ./ std(s)
    s .-= minimum(s)
    s ./= maximum(s)

    N = length(s)
    Θ = generate_basis_functions(N)'

    if method == "lasso"
        reg_meth = lasso()
        return compute_spectrum_according_to_threshold(reg_meth, s, sparse(Θ), ρ_thres, tol, max_iter, verbose)
    else
        reg_meth = STLS()
        return compute_spectrum_according_to_threshold(reg_meth, s, sparse(Θ), ρ_thres, tol, max_iter, verbose)
    end
end

"""
    Determine the right LASSO-regularization parameter with respect to ρ_thres & tol
    using the Newton-method.
"""
function compute_spectrum_according_to_threshold(reg_meth::lasso, s::Vector, Θ::Union{SparseMatrixCSC, AbstractMatrix}, ρ_thres::Real, tol::Real, max_iter::Integer, verbose::Bool)

    abs_tol = 1e-6
    # initial Lambda-step for estimating an upper bound for λ
    lambda_step = 0.5
    lambda_max, lambda_min, y_act, ρ_act = find_lambda_max(s, Θ, lambda_step, ρ_thres)

    # bisection search
    for i = 1:max_iter
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
        # check whether max iterations or tolerance-level reached
        if i == max_iter
            if ρ_act > 1 - abs_tol
                if verbose
                    println("Algorithm stopped due to maximum number of λ's were tried without convergence to the specified `ρ_thres`. Perfect deomposition achieved.")
                end
                debias_coefficients!(y_act, s, Θ) # coefficient de-biasing
                spectrum_i = pool_frequencies(y_act, length(s))
                spectrum_i = spectrum_i ./ sum(spectrum_i) # normalization
                spectrum, y = compute_spectrum_according_to_actual_spectrum(reg_meth, spectrum_i, s, Θ, actual_lambda)
                return spectrum, cor(vec(regenerate_signal(Θ, y)), s)
            else
                if verbose
                    println("Algorithm stopped due to maximum number of λ's were tried without convergence to the specified `ρ_thres`")
                end
                debias_coefficients!(y_act, s, Θ) # coefficient de-biasing
                spec = pool_frequencies(y_act, length(s))
                spec = spec ./ sum(spec) # normalization
                return spec , ρ_act
            end
            break
        elseif abs(ρ_act - ρ_thres) <= tol
            debias_coefficients!(y_act, s, Θ) # coefficient de-biasing
            spec = pool_frequencies(y_act, length(s))
            spec = spec ./ sum(spec) # normalization
            return spec , ρ_act
        end
    end
end

"""
    Determine the right STLS-regularization parameter with respect to ρ_thres & tol
    using the Newton-method.
"""
function compute_spectrum_according_to_threshold(reg_meth::STLS, s::Vector, Θ::AbstractMatrix, ρ_thres::Real, tol::Real, max_iter::Integer, verbose::Bool)

    abs_tol = 1e-6
    # lambda_max/ lambda_min correspond to the maximum/minimum value in the coeffs of least squares
    #y_act = pinv(Θ)*s     # least squares
    #y_act = qr(Θ, Val(true))\s 
    #y_act = qr(Θ; ordering = Int32(1))\s   # this is for a sparse-type Θ
    y_act = lsqr(Θ,s)
    lambda_maxx = maximum(y_act)
    lambda_max = maximum(y_act)
    lambda_min = minimum(y_act)

    # bisection search
    for i = 1:max_iter
        # try new lambda
        actual_lambda = lambda_min + (lambda_max - lambda_min)/2
        # make the regression with specific lambda
        y_act[:] = stls(s, Θ, actual_lambda)
        # check whether the regenerated signal matches with the given threshold
        rr = cor(vec(regenerate_signal(Θ, y_act)), s)

        if isnan(rr)
            lambda_max = actual_lambda
        elseif rr < ρ_thres
            lambda_max = actual_lambda
            ρ_act = rr
        elseif rr > ρ_thres
            lambda_min = actual_lambda
            ρ_act = rr
        end

        # check whether max iterations or tolerance-level reached
        if i == max_iter
            if ρ_act > 1 - abs_tol
                if verbose
                    println("Algorithm stopped due to maximum number of λ's were tried without convergence to the specified `ρ_thres`. Perfect deomposition achieved.")
                end
                spectrum_i = pool_frequencies(y_act, length(s))
                spectrum_i = spectrum_i ./ sum(spectrum_i) # normalization
                spectrum, y = compute_spectrum_according_to_actual_spectrum(reg_meth, spectrum_i, s, Θ, actual_lambda, lambda_maxx)
                return spectrum, cor(vec(regenerate_signal(Θ, y)), s)
            else
                if verbose
                    println("Algorithm stopped due to maximum number of λ's were tried without convergence to the specified `ρ_thres`")
                end
                spec = pool_frequencies(y_act, length(s))
                spec = spec ./ sum(spec) # normalization
                return spec , ρ_act
            end
            break
        elseif abs(ρ_act - ρ_thres) <= tol
            spec = pool_frequencies(y_act, length(s))
            spec = spec ./ sum(spec) # normalization
            return spec , ρ_act
        end
    end
end


"""
    Determine the right LASSO-regularization parameter with respect to ρ_thres & tol
    using the Newton-method.
"""
function compute_spectrum_according_to_actual_spectrum(reg_meth::lasso, spectrum_i::Vector, s::Vector,  Θ::Union{SparseMatrixCSC, AbstractMatrix}, λ_max::Real)
    abs_tol = 1e-6
    max_iter = 10 # precision
    λ_min = 0
    y_act = zeros(size(Θ,2))
    spectrum = zeros(size(spectrum_i))
    # bisection search
    for i = 1:max_iter
        # try new lambda
        actual_λ = λ_min + (λ_max - λ_min)/2
        # make the regression with specific lambda
        path = glmnet(Θ, @view s[:]; lambda = [actual_λ])
        y_act[:] = path.betas
        debias_coefficients!(y_act, s, Θ) # coefficient de-biasing
        spectrum[:] = pool_frequencies(y_act, length(s))
        spectrum = spectrum ./ sum(spectrum) # normalization
        # check whether the spectrum matches with the initial spectrum (input)
        rr = cor(spectrum, spectrum_i)

        # pick the new bisection interval
        if rr > 1-abs_tol
            λ_max = actual_λ
        elseif rr < 1-abs_tol
            λ_min = actual_λ
        end
    end
    return spectrum, y_act
end

"""
    Determine the right STLS-regularization parameter with respect to ρ_thres & tol
    using the Newton-method.
"""
function compute_spectrum_according_to_actual_spectrum(reg_meth::STLS, spectrum_i::Vector, s::Vector,  Θ::Union{SparseMatrixCSC, AbstractMatrix}, λ_min::Real, λ_max::Real)

    @assert λ_min < λ_max "λ_min must be smaller than λ_max."
    abs_tol = 1e-6
    max_iter = 10 # precision
    y_act = zeros(size(Θ,2))
    spectrum = zeros(size(spectrum_i))
    # bisection search
    for i = 1:max_iter
        # try new lambda
        actual_λ = λ_min + (λ_max - λ_min)/2
        # make the regression with specific lambda
        y_act[:] = stls(s, Θ, actual_λ)
        spectrum[:] = pool_frequencies(y_act, length(s))
        spectrum = spectrum ./ sum(spectrum) # normalization
        # check whether the spectrum matches with the initial spectrum (input)
        rr = cor(spectrum, spectrum_i)

        # pick the new bisection interval
        if rr > 1-abs_tol
            λ_max = actual_λ
        elseif rr < 1-abs_tol
            λ_min = actual_λ
        end
    end
    return spectrum, y_act
end


"""
    stls(s::Vector, Θ::AbstractMatrix, lambda::Real; iterations::Integer=10) → coefficients
    
    Sequential Thresholded Least Squares sparse regression method
"""
function stls(s::Vector, Θ::Union{SparseMatrixCSC, AbstractMatrix}, lambda::Real)
    max_iter = 10
    tol = 1e-5

    coeffs = lsqr(Θ,s) # initial guess least squares
    biginds_old = deepcopy(coeffs)

    k = 1
    while k < max_iter
        smallinds = (abs.(coeffs).<lambda)  # find small coefficients 
        coeffs[smallinds] .= 0  # threshold these coeffs
        biginds = .! smallinds
        # check convergence
        if maximum(abs.(biginds .- biginds_old)) < tol
            break
        end
        coeffs[biginds] .= lsqr(Θ[:,biginds],s) # regress onto remaining terms
        biginds_old = biginds
        k += 1
    end
    return coeffs
end

# make a least-squares regression on the non-zero coefficients from the sparse regression
function debias_coefficients!(coeffs::Vector, s::Vector, Θ::Union{SparseMatrixCSC, AbstractMatrix})
    non_zero_idx = findall(x->x!=0, coeffs)
    de_biased_coeffs = lsqr(Θ[:,non_zero_idx],s)
    coeffs[non_zero_idx] .= de_biased_coeffs
end

# regenerate a decomposed signal from basis functions and its coefficients
function regenerate_signal(Θ::Union{SparseMatrixCSC, AbstractMatrix}, coefs::Union{SparseVector, Vector, SubArray, CompressedPredictorMatrix})
    @assert size(Θ,2) == length(coefs)
    return Θ*coefs
end

# pool the same frequencies
function pool_frequencies(y::Vector, N::Integer)
    y = abs.(y)
    M = Int(ceil(N/2))
    spectrum = zeros(M)
    cnt = 1
    for i = 1:M
        spectrum[i] = sum(y[cnt:cnt+i-1])
        cnt += i
    end
    return spectrum
end

# estimate a maximum λ-value for which the correlation coefficient form the re-
# generated signal and the input signal falls below `ρ_thres` or is NaN.
function find_lambda_max(s::Vector, Θ::Union{SparseMatrixCSC, AbstractMatrix}, lambda_step::Real, ρ_thres::Real)
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

    Generate a matrix `basis` with all possible `sum(1:ceil(N/2))` spike-basis functions.
    These functions are not independent, since we account for time shifts. Thus,
    `basis` is a set of linear-dependent basis- functions.
"""
function generate_basis_functions(N::Int)
    @assert N > 0
    M = sum(1:Int(ceil(N/2)))

    num_of_basis_functions = M
    basis = zeros(num_of_basis_functions, N)
    cnt = 1
    for i = 1:Int(ceil(N/2))
       basis[cnt:cnt+i-1,:] = create_single_basis_function(N, i)
       cnt += i
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
