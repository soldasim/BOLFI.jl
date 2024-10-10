
# TODO docs
mutable struct LikelihoodStd{
    I<:Union{IterLimit, NoLimit},
} <: BolfiTermCond
    iter_limit::I
    min_iters::Int
    tau::Int
    max_rmsd::Float64
    last_iter::Union{Nothing, BolfiProblem}
    history
end
function LikelihoodStd(;
    max_iters = nothing,
    min_iters = 10,
    tau = 10,
    max_rmsd = 0.1,
)
    iter_limit = isnothing(max_iters) ? NoLimit() : IterLimit(max_iters)
    return LikelihoodStd(iter_limit, min_iters, tau, max_rmsd, nothing, nothing)
end

function (cond::LikelihoodStd)(bolfi::BolfiProblem{Nothing})
    cond.iter_limit(bolfi.problem) || return false

    prev_bolfi = cond.last_iter
    cond.last_iter = deepcopy(bolfi)

    isnothing(prev_bolfi) && return true

    new_datum = bolfi.problem.data.X[:,end]
    max_ll_std = calculate(cond, prev_bolfi, new_datum)

    up_history(cond, max_ll_std)

    iters = length(cond.history)
    (iters < cond.min_iters) && return true

    return rmsd(cond) > cond.max_rmsd
end
function (cond::LikelihoodStd)(bolfi::BolfiProblem{Matrix{Bool}})
    cond.iter_limit(bolfi.problem) || return false

    prev_bolfi = cond.last_iter
    cond.last_iter = bolfi

    isnothing(prev_bolfi) && return true

    new_datum = bolfi.problem.data.X[:,end]
    max_ll_std = calculate.(Ref(cond), get_subset.(Ref(prev_bolfi), eachcol(prev_bolfi.y_sets)), Ref(new_datum))
    
    up_history(cond, max_ll_std)

    iters = length(cond.history)
    (iters < cond.min_iters) && return true
    
    # TODO
    throw(ErrorException("NOT IMPLEMENTED"))
end

function rmsd(cond::LikelihoodStd)
    return rmsd(cond.history[end-cond.tau+1:end])
end

function rmsd(vals::AbstractVector{<:Real})
    deltas = vals .- mean(vals)
    deltas ./= mean(vals) # normalize
    rmsd = sqrt(mean(deltas .^ 2))
    return rmsd
end

function calculate(cond::LikelihoodStd, bolfi::BolfiProblem, new_datum::AbstractVector{<:Real})
    var = posterior_variance(bolfi)
    return sqrt(var(new_datum))
end

function up_history(cond::LikelihoodStd, cost::Union{Float64, Vector{Float64}})
    if isnothing(cond.history)
        cond.history = [cost]
    else
        push!(cond.history, cost)
    end
end
