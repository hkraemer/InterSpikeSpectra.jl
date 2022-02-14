# Main functionality of the spike train decomposition and the obtained inter
# spike spectrum
"""
    inter_spike_spectrum(s::Vector; kwargs...) → spectrum, ρ

    Compute a spike `spectrum` from the a signal `s`, using a LASSO optimization.
    `s` will get normalized to the unity interval `s ∈ [0 1]` internally. Second
    output `ρ` is the linear Pearson correlation coefficient between the true
    input `s` and the regenerated decomposed signal.

    Keyword arguments:
    `ρ_thres = 0.99`: The agreement of the regenerated decomposed signal with the
                      true signal. This depends on the LASSO regularization parameter
                      `λ`. `λ` gets adjusted automatically with respect to `ρ_thres`.
    `tol = 1e-3`: Allowed tolerance between `ρ_thres` and `ρ`.
    `max_iter = 15`: Determines after how many tried Lambdas the algorithm stopps.
    `verbose::Bool=true`: If true, warning messages enabled.
"""
function inter_spike_spectrum3(s::Vector{T}; ρ_thres::Real = 0.99, tol::Real=1e-3, max_iter::Integer=15, verbose::Bool=true) where {T}
    @assert 0.8 <= ρ_thres <= 1 "Optional input `ρ_thres` must be a value in the interval [0.8, 1]"
    @assert 1e-5 <= tol <= 1 "Optional input `tol` must be a value in the interval [1e-5, 1]"
    @assert 1 < max_iter <= 20 "Optional input `max_iter` must be an integer in the interval (1, 20]."

    # normalization
    s = (s.-mean(s)) ./ std(s)
    s .-= minimum(s)
    s ./= maximum(s)

    N = length(s)
    Θ = generate_basis_functions2(N)'

    #display("Size of Θ is: $(size(sparse(Θ)))")

    return compute_spectrum_according_to_threshold3(s, sparse(Θ), ρ_thres, tol, max_iter, verbose)
end

"""
    hallo
"""
function inter_spike_spectrum2(s::Vector{T}; ρ_thres::Real = 0.99, tol::Real=1e-3, max_iter::Integer=15, verbose::Bool=true) where {T}
    @assert 0.8 <= ρ_thres <= 1 "Optional input `ρ_thres` must be a value in the interval [0.8, 1]"
    @assert 1e-5 <= tol <= 1 "Optional input `tol` must be a value in the interval [1e-5, 1]"
    @assert 1 < max_iter <= 20 "Optional input `max_iter` must be an integer in the interval (1, 20]."

    # normalization
    s = (s.-mean(s)) ./ std(s)
    s .-= minimum(s)
    s ./= maximum(s)

    N = length(s)

    non_unique_idx  = non_unique_lines(N)
    Θ = generate_basis_functions2(N)
    size_full = size(Θ,1)
    full_idx = collect(1:size(Θ,1))
    setdiff!(full_idx, non_unique_idx)

    #display("Size of Θ is: $(size(sparse(view(Θ,full_idx,:)')))")

    return compute_spectrum_according_to_threshold2(s, sparse(view(Θ,full_idx,:)'), ρ_thres, tol, max_iter, full_idx, size_full, verbose)
end

"""
    hallo 2
"""
function inter_spike_spectrum(s::Vector{T}; ρ_thres::Real = 0.99, tol::Real=1e-3, max_iter::Integer=15, verbose::Bool=true) where {T}
    @assert 0.8 <= ρ_thres <= 1 "Optional input `ρ_thres` must be a value in the interval [0.8, 1]"
    @assert 1e-5 <= tol <= 1 "Optional input `tol` must be a value in the interval [1e-5, 1]"
    @assert 1 < max_iter <= 20 "Optional input `max_iter` must be an integer in the interval (1, 20]."

    # normalization
    s = (s.-mean(s)) ./ std(s)
    s .-= minimum(s)
    s ./= maximum(s)

    N = length(s)
    Θ = generate_basis_functions(N)'

    #display("Size of Θ is: $(size(sparse(Θ)))")

    return compute_spectrum_according_to_threshold(s, sparse(Θ), ρ_thres, tol, max_iter, verbose)
end

"""
    Determine the right regularization parameter with respect to ρ_thres & tol
    using the Newton-method.
"""
function compute_spectrum_according_to_threshold3(s::Vector, Θ::SparseMatrixCSC, ρ_thres::Real, tol::Real, max_iter::Integer, verbose::Bool)
    abs_tol = 1e-6
    # initial Lambda-step for estimating an upper bound for λ
    lambda_step = 0.5
    lambda_max, lambda_min, y_act, ρ_act = find_lambda_max(s, Θ, lambda_step, ρ_thres)
    actual_lambda = 0
    # bisection search
    for i = 1:max_iter
        # check whether max iterations or tolerance-level reached
        if i == max_iter
            if ρ_act > 1 - abs_tol
                if verbose
                    println("Algorithm stopped due to maximum number of λ's were tried without convergence to the specified `ρ_thres`. Perfect deomposition achieved.")
                end
                spectrum_i = pool_frequencies3(y_act, length(s))
                spectrum_i = spectrum_i ./ sum(spectrum_i)
                spectrum, y = compute_spectrum_according_to_actual_spectrum3(spectrum_i, s, Θ, actual_lambda)
                return spectrum, cor(vec(regenerate_signal(Θ, y)), s)
            else
                if verbose
                    println("Algorithm stopped due to maximum number of λ's were tried without convergence to the specified `ρ_thres`")
                end
                spec = pool_frequencies3(y_act, length(s))
                spec = spec ./ sum(spec)
                return spec , ρ_act
            end
            break
        elseif abs(ρ_act - ρ_thres) <= tol
            spec = pool_frequencies3(y_act, length(s))
            spec = spec ./ sum(spec)
            return spec , ρ_act
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

function compute_spectrum_according_to_threshold2(s::Vector, Θ::SparseMatrixCSC, ρ_thres::Real, tol::Real, max_iter::Integer, unique_idx::Vector, size_full::Int, verbose::Bool)
    abs_tol = 1e-6
    # initial Lambda-step for estimating an upper bound for λ
    lambda_step = 0.5
    lambda_max, lambda_min, y_act, ρ_act = find_lambda_max(s, Θ, lambda_step, ρ_thres)
    actual_lambda = 0
    # bisection search
    for i = 1:max_iter
        # check whether max iterations or tolerance-level reached
        if i == max_iter
            if ρ_act > 1 - abs_tol
                if verbose
                    println("Algorithm stopped due to maximum number of λ's were tried without convergence to the specified `ρ_thres`. Perfect deomposition achieved.")
                end
                spectrum_i = pool_frequencies2(y_act, length(s), unique_idx, size_full)
                spectrum_i = spectrum_i ./ sum(spectrum_i)
                spectrum, y = compute_spectrum_according_to_actual_spectrum2(spectrum_i, s, Θ, actual_lambda, unique_idx, size_full)
                return spectrum, cor(vec(regenerate_signal(Θ, y)), s)
            else
                if verbose
                    println("Algorithm stopped due to maximum number of λ's were tried without convergence to the specified `ρ_thres`")
                end
                spec = pool_frequencies2(y_act, length(s), unique_idx, size_full)
                spec = spec ./ sum(spec)
                return spec , ρ_act
            end
            break
        elseif abs(ρ_act - ρ_thres) <= tol
            spec = pool_frequencies2(y_act, length(s), unique_idx, size_full)
            spec = spec ./ sum(spec)
            return spec , ρ_act
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

function compute_spectrum_according_to_threshold(s::Vector, Θ::SparseMatrixCSC, ρ_thres::Real, tol::Real, max_iter::Integer, verbose::Bool)
    abs_tol = 1e-6
    # initial Lambda-step for estimating an upper bound for λ
    lambda_step = 0.5
    lambda_max, lambda_min, y_act, ρ_act = find_lambda_max(s, Θ, lambda_step, ρ_thres)
    actual_lambda = 0
    # bisection search
    for i = 1:max_iter
        # check whether max iterations or tolerance-level reached
        if i == max_iter
            if ρ_act > 1 - abs_tol
                if verbose
                    println("Algorithm stopped due to maximum number of λ's were tried without convergence to the specified `ρ_thres`. Perfect deomposition achieved.")
                end
                spectrum_i = pool_frequencies(y_act, length(s))
                spectrum_i = spectrum_i ./ sum(spectrum_i)
                spectrum, y = compute_spectrum_according_to_actual_spectrum(spectrum_i, s, Θ, actual_lambda)
                return spectrum, cor(vec(regenerate_signal(Θ, y)), s)
            else
                if verbose
                    println("Algorithm stopped due to maximum number of λ's were tried without convergence to the specified `ρ_thres`")
                end
                spec = pool_frequencies(y_act, length(s))
                spec = spec ./ sum(spec)
                return spec , ρ_act
            end
            break
        elseif abs(ρ_act - ρ_thres) <= tol
            spec = pool_frequencies(y_act, length(s))
            spec = spec ./ sum(spec)
            return spec , ρ_act
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

"""
    Determine the right regularization parameter with respect to ρ_thres & tol
    using the Newton-method.
"""
function compute_spectrum_according_to_actual_spectrum3(spectrum_i::Vector, s::Vector,  Θ::SparseMatrixCSC, λ_max::Real)
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
        spectrum[:] = pool_frequencies3(y_act, length(s))
        spectrum = spectrum ./ sum(spectrum)
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
function compute_spectrum_according_to_actual_spectrum2(spectrum_i::Vector, s::Vector,  Θ::SparseMatrixCSC, λ_max::Real, unique_idx::Vector, size_full::Integer)
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
        spectrum[:] = pool_frequencies2(y_act, length(s), unique_idx, size_full)
        spectrum = spectrum ./ sum(spectrum)
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
function compute_spectrum_according_to_actual_spectrum(spectrum_i::Vector, s::Vector,  Θ::SparseMatrixCSC, λ_max::Real, unique_idx::Vector, size_full::Integer)
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
        spectrum[:] = pool_frequencies(y_act, length(s), unique_idx, size_full)
        spectrum = spectrum ./ sum(spectrum)
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

# regenerate a decomposed signal from basis functions and its coefficients
function regenerate_signal(Θ::SparseMatrixCSC, coefs::CompressedPredictorMatrix)
    @assert size(Θ,2) == length(coefs) "Number of basis functions must match number of coefficients"
    return Θ*coefs
end

# pool the same frequencies
function pool_frequencies3(y::Vector, N::Integer)
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
function pool_frequencies2(yy::Vector, N::Integer, unique_idx::Vector, size_full::Integer)
    y = zeros(size_full)
    y[unique_idx] .= abs.(yy)
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
function pool_frequencies(y::Vector, N::Integer)
    y = abs.(y)
    M = Int(ceil(N/2))
    spectrum = zeros(M)
    cnt = 1
    for i = 1:M
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
    M = sum(1:Int(ceil(N/2)))

    num_of_basis_functions = M
    basis = zeros(num_of_basis_functions, N)
    cnt = 1
    for i = 1:Int(ceil(N/2))
       basis[cnt:cnt+i-1,:] = create_single_basis_function(N, i);
       cnt = cnt + i;
    end
    return basis
end
function generate_basis_functions2(N::Int)
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

# Compute all line indices, which correspond to non-unique lines in the matrix,
# which would store all possible `sum(1:N)` spike-basis functions.
function non_unique_lines(N::Integer)
    @assert N > 0
    idx = Int[]
    if N == 1 || N == 2
        return idx
    end
    M = floor(N/2 +1)
    LPD = N  # init lowest possible spike period
    for shift = 1:N-2
        # lowest possible spike period for this shift
        if M-(N-shift) == 1
            # this causal is needed for even and uneven N
            if LPD-shift == 1
                nothing
            elseif LPD-shift == 0
                LPD +=1
            end
        elseif N-shift >= M
            LPD = N-shift
        elseif N-shift < M
            LPD += 1
        end
        for i = 1:(N-LPD)
            push!(idx, index_in_basis_function_matrix(LPD+i)+shift)
        end
    end
    return idx
end

# For an inter spike period T, compute the line index in the corresponding matrix,
# which would store all possible `sum(1:N)` spike-basis functions.
function index_in_basis_function_matrix(T::Integer)
    return (T*((1+T)/2) - (T-1))
end

# regenerate a decomposed signal from basis functions and its coefficients
function regenerate_signal(Θ::Union{SparseMatrixCSC, Matrix}, coefs::Union{SparseVector, Vector, SubArray})
    @assert size(Θ,2) == length(coefs)
    return Θ*coefs
end
