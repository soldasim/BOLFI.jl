using KernelFunctions # TODO

# TODO docs
"""
Calculate MMD between approximate and expected posterior.
"""
mutable struct MaximumMeanDiscrepancy{
    I<:Union{IterLimit, NoLimit},
} <: BolfiTermCond
    iter_limit::I
    samples::Int64
    gauss_opt::GaussMixOptions
    kernel::BOSS.Kernel
    target_mmd::Float64
    history
end
function MaximumMeanDiscrepancy(;
    max_iters = nothing,
    samples,
    gauss_opt,
    kernel = GaussianKernel(),
    target_mmd,
)
    iter_limit = isnothing(max_iters) ? NoLimit() : IterLimit(max_iters)
    return MaximumMeanDiscrepancy(
        iter_limit,
        samples,
        gauss_opt,
        kernel,
        target_mmd,
        nothing,
    )
end

function (cond::MaximumMeanDiscrepancy)(bolfi::BolfiProblem{Nothing})
    cond.iter_limit(bolfi.problem) || return false

    mmd = calculate(cond, bolfi)
    up_history(cond, mmd)
    
    return mmd > cond.target_mmd
end
function (cond::MaximumMeanDiscrepancy)(bolfi::BolfiProblem{Matrix{Bool}})
    cond.iter_limit(bolfi.problem) || return false
    
    mmds = calculate.(Ref(cond), get_subset.(Ref(bolfi), eachcol(bolfi.y_sets)))
    up_history(cond, mmds)
    
    # TODO
    throw(ErrorException("NOT IMPLEMENTED"))
end

function calculate(cond::MaximumMeanDiscrepancy, bolfi::BolfiProblem)
    domain = bolfi.problem.domain

    # TODO
    post_1 = approx_posterior(bolfi)
    post_2 = posterior_mean(bolfi)
    # gp_post = model_posterior(bolfi.problem)
    # post_1 = approx_posterior(gp_bound(gp_post, -1.), bolfi.x_prior, bolfi.std_obs)
    # post_2 = approx_posterior(gp_bound(gp_post, +1.), bolfi.x_prior, bolfi.std_obs)

    q_1 = approx_by_gauss_mix(post_1, domain, cond.gauss_opt)
    q_2 = approx_by_gauss_mix(post_2, domain, cond.gauss_opt)

    any(isnothing.((q_1, q_2))) && return nothing

    # # private samples
    # xs_1 = _sample_samples_mmd(q_1, cond.samples, domain)
    # xs_2 = _sample_samples_mmd(q_2, cond.samples, domain)
    # ws_1 = _calculate_weights_mmd(post_1, q_1, xs_1)
    # ws_2 = _calculate_weights_mmd(post_2, q_2, xs_2)

    # return mmd(xs_1, xs_2, ws_1, ws_2, cond.kernel)

    # shared samples
    q = MixtureModel([q_1, q_2], [0.5, 0.5])
    xs = _sample_samples_mmd(q, cond.samples, domain)
    ws_1 = _calculate_weights_mmd(post_1, q, xs)
    ws_2 = _calculate_weights_mmd(post_2, q, xs)

    return mmd(xs, xs, ws_1, ws_2, cond.kernel)
end

# lowest variance estimator
# biased estimator
# always >= 0
function mmd(xs, ys, ws, vs, k)
    @assert isapprox(sum(ws), 1.; atol=1e-8)
    @assert isapprox(sum(vs), 1.; atol=1e-8)

    Kx = pairwise(k, xs, xs)
    Ky = pairwise(k, ys, ys)
    Kxy = pairwise(k, xs, ys)

    # value = mean(Kx) + mean(Ky) - 2*mean(Kxy)
    value = (ws' * Kx * ws) + (vs' * Ky * vs) - 2 * (ws' * Kxy * vs)
    return sqrt(value)
end

function _sample_samples_mmd(proposal_dist, samples, domain::Domain)
    xs = rand(proposal_dist, samples)
    xs = [x for x in eachcol(xs) if BOSS.in_domain(x, domain)]
    return xs
end

function _calculate_weights_mmd(true_post, proposal_dist, xs)
    ws = true_post.(xs) ./ pdf.(Ref(proposal_dist), xs)
    ws ./= sum(ws)
    return ws
end

function up_history(cond::MaximumMeanDiscrepancy, cost::Union{Float64, Vector{Float64}})
    if isnothing(cond.history)
        cond.history = [cost]
    else
        push!(cond.history, cost)
    end
end
