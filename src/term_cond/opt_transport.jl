
# TODO docs
mutable struct OptTransport{
    I<:Union{IterLimit, NoLimit},
} <: BolfiTermCond
    iter_limit::I
    min_iters::Int64
    tau::Int64
    max_rmsd::Float64
    samples::Int64
    gauss_opt::GaussMixOptions
    metric::PreMetric
    epsilon::Float64
    sinkhorn_tol::Float64
    sinkhorn_iters::Int64
    history
end
function OptTransport(;
    max_iters = nothing,
    min_iters = 10,
    tau = 10,
    max_rmsd = 0.1,
    samples,
    gauss_opt,
    metric = SqEuclidean(),
    epsilon = 1.,
    sinkhorn_tol = 1e-2,
    sinkhorn_iters = 1000,
)
    iter_limit = isnothing(max_iters) ? NoLimit() : IterLimit(max_iters)
    return OptTransport(
        iter_limit,
        min_iters,
        tau,
        max_rmsd,
        samples,
        gauss_opt,
        metric,
        epsilon,
        sinkhorn_tol,
        sinkhorn_iters,
        nothing,
    )
end

function (cond::OptTransport)(bolfi::BolfiProblem{Nothing})
    cond.iter_limit(bolfi.problem) || return false

    cost = calculate(cond, bolfi)
    up_history(cond, cost)

    iters = length(cond.history)
    (iters < cond.min_iters) && return true
    
    # history_ = [isnothing(c) ? Inf : c for c in cond.history]
    # best = argmin(history_)
    # return ((iters - best) / iters) < (1 / cond.tau)

    return rmsd(cond) > cond.max_rmsd
end
function (cond::OptTransport)(bolfi::BolfiProblem{Matrix{Bool}})
    cond.iter_limit(bolfi.problem) || return false
    
    costs = calculate.(Ref(cond), get_subset.(Ref(bolfi), eachcol(bolfi.y_sets)))
    up_history(cond, costs)

    iters = length(cond.history)
    (iters < cond.min_iters) && return true
    
    # history_ = hcat(cond.history...)
    # history_[isnothing.(history_)] .= Inf
    # best = [argmin(h_) for h_ in eachrow(history_)]
    # return any( ((iters .- best) ./ iters) .< (1 / cond.tau) )

    # TODO
    throw(ErrorException("NOT IMPLEMENTED"))
end

function rmsd(cond::OptTransport)
    history_ = [isnothing(c) ? Inf : c for c in cond.history[end-cond.tau+1:end]]
    return rmsd(history_)
end

function rmsd(vals::AbstractVector{<:Real})
    deltas = vals .- mean(vals)
    deltas ./= mean(vals) # normalize
    rmsd = sqrt(mean(deltas .^ 2))
    return rmsd
end

function calculate(cond::OptTransport, bolfi::BolfiProblem)
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

    # ws_1, xs_1 = _sample_samples_ot(post_1, q_1, cond.samples, domain)
    # ws_2, xs_2 = _sample_samples_ot(post_2, q_2, cond.samples, domain)
    # C = pairwise(cond.metric, vec(xs_1), vec(xs_2))
    
    # Shared samples
    q = MixtureModel([q_1, q_2], [1/2, 1/2])
    xs = _sample_samples_ot(q, cond.samples, domain)
    ws_1 = _calculate_weights_ot(post_1, q, xs)
    ws_2 = _calculate_weights_ot(post_2, q, xs)
    C = pairwise(cond.metric, vec(xs), vec(xs))

    # # Separate samples
    # xs_1 = _sample_samples_ot(q_1, cond.samples, domain)
    # xs_2 = _sample_samples_ot(q_2, cond.samples, domain)
    # ws_1 = _calculate_weights_ot(post_1, q_1, xs_1)
    # ws_2 = _calculate_weights_ot(post_2, q_2, xs_2)
    # C = pairwise(cond.metric, vec(xs_1), vec(xs_2))

    # T = sinkhorn(ws_1, ws_2, C, cond.epsilon; maxiter=cond.sinkhorn_iters, tol=cond.sinkhorn_tol)
    cost = sinkhorn2(ws_1, ws_2, C, cond.epsilon; maxiter=cond.sinkhorn_iters, tol=cond.sinkhorn_tol)
    return cost
end

function _sample_samples_ot(proposal_dist, samples, domain::Domain)
    xs = rand(proposal_dist, samples)
    xs = [x for x in eachcol(xs) if BOSS.in_domain(x, domain)]
    return xs
end

function _calculate_weights_ot(true_post, proposal_dist, xs)
    ws = true_post.(xs) ./ pdf.(Ref(proposal_dist), xs)
    ws ./= sum(ws)
    return ws
end

# function _sample_samples_ot(true_post, proposal_dist, samples::Int, domain::Domain)
#     xs = rand(proposal_dist, samples)
#     xs = [x for x in eachcol(xs) if BOSS.in_domain(x, domain)]
#     ws = true_post.(xs) ./ pdf.(Ref(proposal_dist), xs)
#     ws ./= sum(ws)
#     return ws, xs
# end

function up_history(cond::OptTransport, cost::Union{Float64, Vector{Float64}})
    if isnothing(cond.history)
        cond.history = [cost]
    else
        push!(cond.history, cost)
    end
end
