
"""
    approx_posterior(::BolfiProblem; kwargs...)

Return the MAP estimation of the unnormalized approximate posterior ``\\hat{p}(y_o|x) p(x)`` as a function of ``x``.

If `normalize=true`, the resulting posterior is approximately normalized.

The posterior is approximated by directly substituting the predictive means of the GPs
as the discrepancies from the true observation and ignoring both the uncertainty of the GPs
due to a lack of data and due to the simulator evaluation noise.

By using `approx_posterior` or `posterior_mean` one controls,
whether to integrate over the uncertainty in the discrepancy estimate.
In addition to that, by providing a `ModelFitter{MAP}` or a `ModelFitter{BI}` to `bolfi!` one controls,
whether to integrate over the uncertainty in the GP hyperparameters.

# Keywords

- `normalize::Bool`: If `normalize` is set to `true`, the evidence ``\\hat{p}(y_o)```
        is estimated by sampling and the normalized approximate posterior ``\\hat{p}(y_o|x) p(x) / \\hat{p}(y_o)``
        is returned instead of the unnormalized one.
- `xs::Union{Nothing, <:AbstractMatrix{<:Real}}`: Can be used to provide a pre-sampled
        set of samples from the parameter prior ``p(x)`` as a column-wise matrix.
        Only has an effect if `normalize == true`.
- `samples::Int`: Controls the number of samples used to estimate the evidence.
        Only has an effect if `normalize == true` and `isnothing(xs)`.

# See Also

[`posterior_mean`](@ref),
[`posterior_variance`](@ref),
[`approx_likelihood`](@ref)
"""
function approx_posterior(bolfi::BolfiProblem; normalize=false, xs=nothing, samples=10_000)
    x_prior = bolfi.x_prior

    approx_like = approx_likelihood(bolfi)
    approx_post(x) = pdf(x_prior, x) * approx_like(x)

    if normalize
        py = evidence(approx_post, x_prior; xs, samples)
        return (x) -> approx_post(x) / py
    else
        return approx_post
    end
end

"""
    approx_likelihood(::BolfiProblem)

Return the MAP estimation of the likelihood ``\\hat{p}(y_o|x)`` as a function of ``x``.

The likelihood is approximated by directly substituting the predictive means of the GPs
as the discrepancies from the true observation and ignoring both the uncertainty of the GPs
due to a lack of data and due to the simulator evaluation noise.

By using `approx_likelihood` or `likelihood_mean` one controls,
whether to integrate over the uncertainty in the discrepancy estimate.
In addition to that, by providing a `ModelFitter{MAP}` or a `ModelFitter{BI}` to `bolfi!` one controls,
whether to integrate over the uncertainty in the GP hyperparameters.

# See Also

[`likelihood_mean`](@ref),
[`likelihood_variance`](@ref),
[`approx_posterior`](@ref)
"""
function approx_likelihood(bolfi::BolfiProblem)
    return approx_likelihood(typeof(bolfi.problem.params), bolfi)
end

function approx_likelihood(::Type{<:UniFittedParams}, bolfi::BolfiProblem)
    gp_post = BOSS.model_posterior(bolfi.problem)
    return approx_likelihood(bolfi.likelihood, bolfi, gp_post)
end
function approx_likelihood(::Type{<:MultiFittedParams}, bolfi::BolfiProblem)
    gp_posts = BOSS.model_posterior(bolfi.problem)
    sample_count = length(gp_posts)
    
    approx_likes = approx_likelihood.(Ref(bolfi.likelihood), Ref(bolfi), gp_posts)
    
    function exp_approx_like(x)
        return mapreduce(l -> l(x), +, approx_likes) / sample_count
    end
end

"""
    posterior_mean(::BolfiProblem; kwargs...)

Return the expectation of the unnormalized posterior ``\\mathbb{E}[\\hat{p}(y_o|x) p(x)]`` as a function of ``x``.

If `normalize=true`, the resulting expected posterior is approximately normalized.

The returned function maps parameters `x` to the expected posterior probability density value
integrated over the uncertainty of the GPs due to a lack of data and due to the simulator evaluation noise.

By using `approx_posterior` or `posterior_mean` one controls,
whether to integrate over the uncertainty in the discrepancy estimate.
In addition to that, by providing a `ModelFitter{MAP}` or a `ModelFitter{BI}` to `bolfi!` one controls,
whether to integrate over the uncertainty in the GP hyperparameters.

# Keywords

- `normalize::Bool`: If `normalize` is set to `true`, the evidence ``\\hat{p}(y_o)```
        is estimated by sampling and the normalized expected posterior ``\\mathbb{E}[\\hat{p}(y_o|x) p(x)]``
        is returned instead of the unnormalized one.
- `xs::Union{Nothing, <:AbstractMatrix{<:Real}}`: Can be used to provide a pre-sampled
        set of samples from the parameter prior ``p(x)`` as a column-wise matrix.
        Only has an effect if `normalize == true`.
- `samples::Int`: Controls the number of samples used to estimate the evidence.
        Only has an effect if `normalize == true` and `isnothing(xs)`.

# See Also

[`approx_posterior`](@ref),
[`posterior_variance`](@ref),
[`likelihood_mean`](@ref)
"""
function posterior_mean(bolfi::BolfiProblem; normalize=false, xs=nothing, samples=10_000)
    x_prior = bolfi.x_prior

    like_mean = likelihood_mean(bolfi)
    post_mean(x) = pdf(x_prior, x) * like_mean(x)

    if normalize
        py = evidence(post_mean, x_prior; xs, samples)
        return (x) -> post_mean(x) / py
    else
        return post_mean
    end
end

"""
    likelihood_mean(::BolfiProblem)

Return the expectation of the likelihood approximation ``\\mathbb{E}[\\hat{p}(y_o|x)]`` as a function of ``x``.

The returned function maps parameters `x` to the expected likelihood probability density value
integrated over the uncertainty of the GPs due to a lack of data and due to the simulator evaluation noise.

By using `approx_likelihood` or `likelihood_mean` one controls,
whether to integrate over the uncertainty in the discrepancy estimate.
In addition to that, by providing a `ModelFitter{MAP}` or a `ModelFitter{BI}` to `bolfi!` one controls,
whether to integrate over the uncertainty in the GP hyperparameters.

# See Also

[`approx_likelihood`](@ref),
[`likelihood_variance`](@ref),
[`posterior_mean`](@ref)
"""
function likelihood_mean(bolfi::BolfiProblem)
    return likelihood_mean(typeof(bolfi.problem.params), bolfi)
end

function likelihood_mean(::Type{<:UniFittedParams}, bolfi::BolfiProblem)
    gp_post = BOSS.model_posterior(bolfi.problem)
    return likelihood_mean(bolfi.likelihood, bolfi, gp_post)
end
function likelihood_mean(::Type{<:MultiFittedParams}, bolfi::BolfiProblem)
    gp_posts = BOSS.model_posterior(bolfi.problem)
    sample_count = length(gp_posts)
    
    like_means = likelihood_mean.(Ref(bolfi.likelihood), Ref(bolfi), gp_posts)
    
    function exp_like_mean(x)
        return mapreduce(l -> l(x), +, like_means) / sample_count
    end
end

"""
    posterior_variance(::BolfiProblem; kwargs...)

Return the variance of the unnormalized posterior ``\\mathbb{V}[\\hat{p}(y_o|x) p(x)]`` as a function of ``x``.

If `normalize=true`, the resulting posterior variance is approximately normalized.

The returned function maps parameters `x` to the variance of the posterior probability density value estimate
caused by the uncertainty of the GPs due to a lack of data and due to the simulator evaluation noise.

By providing a `ModelFitter{MAP}` or a `ModelFitter{BI}` to `bolfi!` one controls,
whether to compute the variance over the uncertainty in the GP hyperparameters as well.

# Keywords

- `normalize::Bool`: If `normalize` is set to `true`, the evidence ``\\hat{p}(y_o)```
        is estimated by sampling and the normalized posterior variance ``\\mathbb{V}[\\hat{p}(y_o|x) p(x) / \\hat{p}(y_o)]``
        is returned instead of the unnormalized one.
- `xs::Union{Nothing, <:AbstractMatrix{<:Real}}`: Can be used to provide a pre-sampled
        set of samples from the parameter prior ``p(x)`` as a column-wise matrix.
        Only has an effect if `normalize == true`.
- `samples::Int`: Controls the number of samples used to estimate the evidence.
        Only has an effect if `normalize == true` and `isnothing(xs)`.

# See Also

[`approx_posterior`](@ref),
[`posterior_mean`](@ref),
[`likelihood_variance`](@ref)
"""
function posterior_variance(bolfi::BolfiProblem; normalize=false, xs=nothing, samples=10_000)
    x_prior = bolfi.x_prior

    like_var = likelihood_variance(bolfi)
    post_var(x) = (pdf(x_prior, x) ^ 2) * like_var(x)

    if normalize
        post_mean = posterior_mean(bolfi)
        py = evidence(post_mean, x_prior; xs, samples)
        py2 = py ^ 2
        return (x) -> post_var(x) / py2
    else
        return post_var
    end
end

"""
    likelihood_variance(::BolfiProblem)

Return the variance of the likelihood approximation ``\\mathbb{V}[\\hat{p}(y_o|x)]`` as a function of ``x``.

The returned function maps parameters `x` to the variance of the likelihood probability density value estimate
caused by the uncertainty of the GPs due to a lack of data and the uncertainty of the simulator
due to the evaluation noise.

By providing a `ModelFitter{MAP}` or a `ModelFitter{BI}` to `bolfi!` one controls,
whether to compute the variance over the uncertainty in the GP hyperparameters as well.

# See Also

[`approx_likelihood`](@ref),
[`likelihood_mean`](@ref),
[`posterior_variance`](@ref)
"""
function likelihood_variance(bolfi::BolfiProblem)
    return likelihood_variance(typeof(bolfi.problem.params), bolfi)
end

function likelihood_variance(::Type{<:UniFittedParams}, bolfi::BolfiProblem)
    gp_post = BOSS.model_posterior(bolfi.problem)
    
    like_mean = likelihood_mean(bolfi.likelihood, bolfi, gp_post)
    sq_like_mean = sq_likelihood_mean(bolfi.likelihood, bolfi, gp_post)

    function like_var(x)
        return sq_like_mean(x) - (like_mean(x) ^ 2)
    end
end
function likelihood_variance(::Type{<:MultiFittedParams}, bolfi::BolfiProblem)
    gp_posts = BOSS.model_posterior(bolfi.problem)
    sample_count = length(gp_posts)
    
    like_means = likelihood_mean.(Ref(bolfi.likelihood), Ref(bolfi), gp_posts)
    sq_like_means = sq_likelihood_mean.(Ref(bolfi.likelihood), Ref(bolfi), gp_posts)

    function like_var(x)
        exp_like = mapreduce(l -> l(x), +, like_means) / sample_count
        exp_sq_like = mapreduce(l -> l(x), +, sq_like_means) / sample_count
        return exp_sq_like - (exp_like ^ 2)
    end
end

"""
    evidence(post, x_prior; kwargs...)

Return the estimated evidence ``\\hat{p}(y_o)``.

# Arguments

- `post`: A function `::AbstractVector{<:Real} -> ::Real`
        representing the posterior ``p(x|y_o)``.
- `x_prior`: A multivariate distribution
        representing the prior ``p(x)``.

# Keywords

- `xs::Union{Nothing, <:AbstractMatrix{<:Real}}`: Can be used to provide a pre-sampled
        set of samples from the `x_prior` as a column-wise matrix.
- `samples::Int`: Controls the number of samples used to estimate the evidence.
        Only has an effect if `isnothing(xs)`.

"""
function evidence(post, x_prior; xs=nothing, samples=10_000)
    ll(x) = post(x) / pdf(x_prior, x)
    isnothing(xs) && (xs = rand(x_prior, samples))
    py = mean((ll(x) for x in eachcol(xs)))
    return py
end
