
module PlotQuality

using BOSS, BOLFI
using Distributions
using Plots

include("toy_problem.jl")


# - - - Callback - - - - -

mutable struct QualityCallback <: BolfiCallback
    qualities::Vector{Float64}
    param_samples::Matrix{Float64}
    true_post::Function
    true_probs::Vector{Float64}
end
function QualityCallback(; x_prior=nothing, samples=2_000, xs=nothing)
    if isnothing(xs)
        xs = rand(x_prior, samples) 
    end

    # true posterior
    function true_post_(x)
        y = ToyProblem.experiment(x; noise_std=zeros(ToyProblem.y_dim))
        ll = pdf(MvNormal(y, ToyProblem.σe_true), ToyProblem.y_obs)
        pθ = pdf(x_prior, x)
        return pθ * ll
    end
    py = evidence(true_post_, x_prior; xs)
    true_post(x) = true_post_(x) / py

    # true probabilities
    true_probs = true_post.(eachcol(xs))

    return QualityCallback(
        Float64[],
        xs,
        true_post,
        true_probs,
    )
end

function (cb::QualityCallback)(bolfi::BolfiProblem; acquisition, options, first, kwargs...)
    exp_post = BOLFI.posterior_mean(bolfi; xs=cb.param_samples)
    exp_probs = exp_post.(eachcol(cb.param_samples))
    diffs = abs.(cb.true_probs .- exp_probs)
    push!(cb.qualities, mean(diffs))
end


# - - - Plot Qualities - - - - -

function plot_approx_quality(bolfi; options, p=nothing, label=nothing, display=true)
    cb = options.callback
    options.info && @info "Plotting ..."
    iters = length(cb.qualities) - 1
    
    if isnothing(p)
        p = plot(; title="approximation quality")
    end
    plot!(p, 0:iters, cb.qualities; label)
    display && Plots.display(p)
    return p
end


end # module PlotQuality
