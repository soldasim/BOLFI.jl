
"""
Termination condition which calculates the Jensen-Shannon divergence
between the approximate and the expected posterior
and terminates once this divergence falls below some target threshold.

# Kwargs
- `max_iters::Union{Nothing, Int}`: Additional limit on the total number of iterations.
- `samples::Int`: How many samples are drawn to approximate the JS divergence.
- `target_div::Float64`: The target divergence. Should be between 0 and log(2).
- `gauss_opt::GaussMixOptions`: Hyperparameters for the Laplace approximation
        used to obtain proposal posterior distributions for sampling.
"""
mutable struct JSDivergence{
    I<:Union{IterLimit, NoLimit},
} <: BolfiTermCond
    iter_limit::I
    target_div::Float64
    samples::Int
    gauss_opt::GaussMixOptions
    history
end
function JSDivergence(;
    max_iters = nothing,
    target_div = 0.02,
    samples = 2000,
    gauss_opt,
)
    iter_limit = isnothing(max_iters) ? NoLimit() : IterLimit(max_iters)
    return JSDivergence(iter_limit, target_div, samples, gauss_opt, nothing)
end

function (cond::JSDivergence)(bolfi::BolfiProblem{Nothing})
    cond.iter_limit(bolfi.problem) || return false
    (bolfi.problem.data isa ExperimentDataPrior) && return true

    div = calculate(cond, bolfi)    
    up_history(cond, div)
    return div > cond.target_div
end

function (cond::JSDivergence)(bolfi::BolfiProblem{Matrix{Bool}})
    cond.iter_limit(bolfi.problem) || return false
    (bolfi.problem.data isa ExperimentDataPrior) && return true

    divs = calculate.(Ref(cond), get_subset.(Ref(bolfi), eachcol(bolfi.y_sets)))
    
    up_history(cond, divs)
    return any(divs .> cond.target_div)
end

function calculate(cond::JSDivergence, bolfi::BolfiProblem)
    # TODO
    post_approx = approx_posterior(bolfi)
    post_expect = posterior_mean(bolfi)
    # post_lb = approx_posterior(gp_bound(model_posterior(bolfi.problem), -1.), bolfi.x_prior, bolfi.std_obs)
    # post_ub = approx_posterior(gp_bound(model_posterior(bolfi.problem), +1.), bolfi.x_prior, bolfi.std_obs)

    q_approx = approx_by_gauss_mix(post_approx, bolfi.problem.domain, cond.gauss_opt)
    q_expect = approx_by_gauss_mix(post_expect, bolfi.problem.domain, cond.gauss_opt)
    # q_lb = approx_by_gauss_mix(post_lb, bolfi.problem.domain, cond.gauss_opt)
    # q_ub = approx_by_gauss_mix(post_ub, bolfi.problem.domain, cond.gauss_opt)

    return jensen_shannon_divergence(post_approx, post_expect, q_approx, q_expect; cond.samples)
    # return jensen_shannon_divergence(post_lb, post_ub, q_lb, q_ub; cond.samples)
end

"""
Compute the Jensen-Shannon divergence of the posteriors `post_1` and `post_2`.

The JS divergence is approximated by sampling from the proposal priors `q1` and `q2`,
where it is assumed that `q1 ≈ post_1` and `q2 ≈ post_2`.
"""
function jensen_shannon_divergence(post_1, post_2, q1::MultivariateDistribution, q2::MultivariateDistribution; samples::Int=2000)
    post_mix(x) = (post_1(x) + post_2(x)) / 2
    
    weight_1(x) = post_1(x) / pdf(q1, x)
    weight_2(x) = post_2(x) / pdf(q2, x)

    val_1(x) = log(post_1(x)) - log(post_mix(x))
    val_2(x) = log(post_2(x)) - log(post_mix(x))

    # sample from proposal distributions
    samples_1 = rand(q1, samples)
    samples_2 = rand(q2, samples)

    # # skip samples with zero weight
    # # (is correct because `lim_{p→0} (p/q) * log(p/r) = 0`)
    # samples_1 = hcat(filter(x -> post_1(x) > 1e-8, eachcol(samples_1))...)
    # samples_2 = hcat(filter(x -> post_2(x) > 1e-8, eachcol(samples_2))...)

    weights_1 = weight_1.(eachcol(samples_1))
    # weights_1 ./= maximum(weights_1) # normalize
    weights_2 = weight_2.(eachcol(samples_2))
    # weights_2 ./= maximum(weights_2) # normalize

    A = (1/2) * sum(weights_1 .* val_1.(eachcol(samples_1))) / sum(weights_1)
    B = (1/2) * sum(weights_2 .* val_2.(eachcol(samples_2))) / sum(weights_2)
    
    # TODO rem
    @show weight_1.(eachcol(samples_1)) |> minimum
    @show weight_2.(eachcol(samples_2)) |> minimum
    @show A, B

    return A + B
end

function up_history(cond::JSDivergence, div::Union{Float64, Vector{Float64}})
    if isnothing(cond.history)
        cond.history = [div]
    else
        push!(cond.history, div)
    end
end
