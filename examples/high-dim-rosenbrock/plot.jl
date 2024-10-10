module Plot

using Plots
using Distributions
using BOSS, BOLFI

include("toy_problem.jl")


# - - - Plotting Callback - - - - -

mutable struct PlotCallback <: BolfiCallback
    prev_state::Union{Nothing, BolfiProblem}
    iters::Int
    plot_each::Int
    display::Bool
    save_plots::Bool
    plot_dir::String
    plot_name::String
    std_obs::Vector{Float64}
    std_sim::Vector{Float64}
    f::Any
    y_obs::Vector{Float64}
    step::Float64
    acquisition::BolfiAcquisition
end
PlotCallback(;
    plot_each::Int = 1,
    display::Bool = true,
    save_plots::Bool = false,
    plot_dir::String = active_dir(),
    plot_name::String = "p",
    std_obs::Vector{Float64},
    std_sim::Vector{Float64},
    f::Any,
    y_obs::Vector{Float64},
    step::Float64 = 0.05,
    acquisition,
) = PlotCallback(nothing, 0, plot_each, display, save_plots, plot_dir, plot_name, std_obs, std_sim, f, y_obs, step, acquisition)

"""
Plots the state in the current iteration.
"""
function (plt::PlotCallback)(bolfi::BolfiProblem; acquisition, options, first, kwargs...)
    if first
        plt.prev_state = deepcopy(bolfi)
        plt.iters += 1
        return
    end
    
    # `iters - 1` because the plot is "one iter behind"
    plot_iter = plt.iters - 1

    if plot_iter % plt.plot_each == 0
        options.info && @info "Plotting ..."
        new_datum = bolfi.problem.data.X[:,end]
        plot_state(plt.prev_state, new_datum; plt, plt.plot_dir, plot_name=plt.plot_name*"_$plot_iter", acquisition=acquisition.acq)
    end
    
    plt.prev_state = deepcopy(bolfi)
    plt.iters += 1
end

"""
Plot the final state after the BO procedure concludes.
"""
function plot_final(plt::PlotCallback; acquisition, options, kwargs...)
    plot_iter = plt.iters - 1
    options.info && @info "Plotting ..."
    plot_name = plt.plot_name*"_$plot_iter"
    plot_state(plt.prev_state, nothing; plt, plt.plot_dir, plot_name, acquisition)
end
plot_final(::Any; kwargs...) = nothing


# - - - Initialization - - - - -

init_plotting(plt::PlotCallback) =
    init_plotting(; save_plots=plt.save_plots, plot_dir=plt.plot_dir)
init_plotting(::Any) = nothing

function init_plotting(; save_plots, plot_dir)
    if save_plots
        if isdir(plot_dir)
            @warn "Directory $plot_dir already exists."
        else
            mkdir(plot_dir)
        end
    end
end


# - - - Plot State - - - - -

function plot_state(bolfi, new_datum; plt, plot_dir=active_dir(), plot_name="p", acquisition)
    # Plots with hyperparams fitted using *all* data! (Does not really matter.)
    p = plot_samples(bolfi; plt, new_datum, acquisition)
    plt.save_plots && savefig(p, plot_dir * '/' * plot_name * ".png")
end

function plot_samples(bolfi; plt, new_datum=nothing, acquisition)
    problem = bolfi.problem
    gp_post = BOSS.model_posterior(problem)

    x_prior = bolfi.x_prior
    bounds = problem.domain.bounds
    @assert all((lb == bounds[1][1] for lb in bounds[1]))
    @assert all((ub == bounds[2][1] for ub in bounds[2]))
    lims = bounds[1][1], bounds[2][1]
    X, Y = problem.data.X, problem.data.Y

    # unnormalized posterior likelihood
    function true_post(a, b)
        x = [a, b]
        y = plt.f(x)

        ll = pdf(MvNormal(y, plt.std_obs), plt.y_obs)
        pθ = pdf(x_prior, x)
        return pθ * ll
    end

    # gp-approximated posterior likelihood
    post_μ = BOLFI.posterior_mean(gp_post, x_prior, plt.std_obs; normalize=false)
    exp_post(a, b) = post_μ([a, b])

    # acquisition
    acq_ = acquisition(bolfi, BolfiOptions())
    function acq(x)
        a = acq_(x)
        if isnan(a) || isinf(a)
            @warn a
        end
        return a
    end

    # - - - PLOT - - - - -
    kwargs = (colorbar=true,)  # TODO

    p1 = plot(; title="true posterior", kwargs...)
    plot_posterior!(p1, true_post; plt, lims, label=nothing)
    plot_samples!(p1, X; plt, new_datum, label=nothing)

    p2 = plot(; title="posterior mean", kwargs...)
    plot_posterior!(p2, exp_post; plt, lims, label=nothing)
    plot_samples!(p2, X; plt, new_datum, label=nothing)

    p3 = plot(; title="abs. value of GP mean", kwargs...)
    plot_posterior!(p3, (a,b) -> abs(gp_post([a,b])[1][1]); plt, lims, label=nothing)
    plot_samples!(p3, X; plt, new_datum, label=nothing)

    p4 = plot(; title="acquisition", kwargs...)
    plot_posterior!(p4, (a,b) -> acq([a,b]); plt, lims, label=nothing)
    plot_samples!(p4, X; plt, new_datum, label=nothing)

    title = ""
    t = plot(; title, framestyle=:none, bottom_margin=-80Plots.px)
    p = plot(t, p1, p2, p3, p4; layout=@layout([°{0.05h}; [° °; ° °;]]), size=(1440, 810))
    plt.display && Plots.display(p)
    return p
end

function plot_posterior!(p, ll; plt, lims)
    grid = lims[1]:plt.step:lims[2]
    contourf!(p, grid, grid, ll)
end

function plot_samples!(p, samples; plt, new_datum=nothing, label)
    scatter!(p, [θ[1] for θ in eachcol(samples)], [θ[2] for θ in eachcol(samples)]; label, color=:green)
    isnothing(new_datum) || scatter!(p, [new_datum[1]], [new_datum[2]]; label=nothing, color=:red)
end


# - - - Plot Parameter Slices - - - - -

function plot_param_slices(bolfi::BolfiProblem; plt, options, samples=20_000, display=true, step=0.05)
    options.info && @info "Plotting ..."
    x_prior = bolfi.x_prior
    
    @assert x_prior isa Product  # individual priors must be independent
    param_samples = rand(x_prior, samples)

    return [plot_param_post(bolfi, i, param_samples; plt) for i in 1:BOLFI.x_dim(bolfi)]
end

function plot_param_post(bolfi, param_idx, param_samples; plt)
    bounds = bolfi.problem.domain.bounds
    x_prior = bolfi.x_prior
    param_range = bounds[1][param_idx]:plt.step:bounds[2][param_idx]
    param_samples_ = deepcopy(param_samples)
    title = "Param $param_idx posterior"

    partial_prior = Product(vcat(x_prior.v[1:param_idx-1], x_prior.v[param_idx+1:end]))
    ws = 1 ./ pdf.(Ref(partial_prior), eachcol(vcat(param_samples_[1:param_idx-1,:], param_samples_[param_idx+1:end,:])))

    # true posterior
    function true_post_(x)
        y = plt.f(x)
        ll = pdf(MvNormal(y, plt.std_obs), plt.y_obs)
        pθ = pdf(x_prior, x)
        return pθ * ll
    end
    py = evidence(true_post_, x_prior; xs=param_samples_)
    true_post(x) = true_post_(x) / py
    
    function true_post_slice(xi)
        param_samples_[param_idx, :] .= xi
        return mean(ws .* true_post.(eachcol(param_samples_)))
    end

    # expected posterior
    exp_post = BOLFI.posterior_mean(bolfi; normalize=true)
    function exp_post_slice(xi)
        param_samples_[param_idx, :] .= xi
        return mean(ws .* exp_post.(eachcol(param_samples_)))
    end

    approx_post = BOLFI.approx_posterior(bolfi; normalize=true)
    function approx_post_slice(xi)
        param_samples_[param_idx, :] .= xi
        return mean(ws .* approx_post.(eachcol(param_samples_)))
    end

    # data
    x = bolfi.problem.data.X[param_idx,:]

    # plot
    p = plot(; title)
    plot!(p, true_post_slice, param_range; label="true posterior")
    plot!(p, exp_post_slice, param_range; label="expected posterior")
    plot!(p, approx_post_slice, param_range; label="approx posterior")
    scatter!(p, x, zeros(length(x)); label="projected data")
    plt.display && Plots.display(p)
    return p
end

end # module Plot
