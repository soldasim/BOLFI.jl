module Plot

using Plots
using Distributions
using BOSS, BOLFI

include("toy_problem.jl")


# - - - Plotting Callback - - - - -

mutable struct PlotCallback <: BolfiCallback
    prev_state::Union{Nothing, BolfiProblem}
    title::String
    iters::Int
    plot_each::Int
    display::Bool
    save_plots::Bool
    plot_dir::String
    plot_name::String
    noise_std_true::Vector{Float64}
    step::Float64
    parallel::Bool
    acq_grid::Union{Nothing, Vector{Float64}}
end
PlotCallback(;
    title = nothing,
    plot_each::Int = 1,
    display::Bool = true,
    save_plots::Bool = false,
    plot_dir::String = active_dir(),
    plot_name::String = "p",
    noise_std_true::Vector{Float64},
    step,
    parallel = true,
    acq_grid = nothing,
) = PlotCallback(nothing, title, 0, plot_each, display, save_plots, plot_dir, plot_name, noise_std_true, step, parallel, acq_grid)

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
        plot_state(plt.prev_state, new_datum; plt.display, plt.save_plots, plt.plot_dir, plot_name=plt.plot_name*"_$plot_iter", plt.noise_std_true, acquisition=acquisition.acq, plt.parallel, plt.acq_grid, plt.title)
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
    plot_state(plt.prev_state, nothing; plt.display, plt.save_plots, plt.plot_dir, plot_name=plt.plot_name*"_$plot_iter", plt.noise_std_true, acquisition, plt.step, plt.parallel, plt.acq_grid, plt.title)
end
plot_final(::Any; kwargs...) = nothing


# - - - Initialization - - - - -

init_plotting(plt::PlotCallback) =
    init_plotting(; save_plots=plt.save_plots, plot_dir=plt.plot_dir)
init_plotting(::Any) = nothing

function init_plotting(; save_plots, plot_dir)
    if save_plots
        if isdir(plot_dir)
            rm(plot_dir, recursive=true)
        end
        mkdir(plot_dir)
    end
end


# - - - Plot State - - - - -

function plot_state(bolfi, new_datum; display=true, save_plots=false, plot_dir=active_dir(), plot_name="p", noise_std_true, acquisition, step=0.05, parallel=true, acq_grid=nothing, title=nothing)
    # Plots with hyperparams fitted using *all* data! (Does not really matter.)
    p = plot_samples(bolfi; new_datum, display, noise_std_true, acquisition, step, parallel, acq_grid, title)
    save_plots && savefig(p, plot_dir * '/' * plot_name * ".png")
end

function plot_samples(bolfi; new_datum=nothing, display=true, noise_std_true, acquisition, step=0.05, parallel=true, acq_grid=nothing, title=nothing)
    problem = bolfi.problem
    gp_post = BOSS.model_posterior(problem)

    x_prior = bolfi.x_prior
    bounds = problem.domain.bounds
    @assert all((lb == bounds[1][1] for lb in bounds[1]))
    @assert all((ub == bounds[2][1] for ub in bounds[2]))
    lims = bounds[1][1], bounds[2][1]
    X, Y = problem.data.X, problem.data.Y

    # unnormalized posterior likelihood `p(d | a, b) * p(a, b) ∝ p(a, b | d)`
    function ll_post(x)
        y = ToyProblem.experiment(x; noise_std=zeros(ToyProblem.y_dim))

        ll = pdf(MvNormal(y, noise_std_true), ToyProblem.y_obs)
        pθ = pdf(x_prior, x)
        return pθ * ll
    end

    # gp-approximated posterior likelihood
    post_mean = BOLFI.posterior_mean(gp_post, x_prior, bolfi.std_obs; normalize=false)
    post_var = BOLFI.posterior_variance(gp_post, x_prior, bolfi.std_obs; normalize=false)

    # acquisition
    acq_samples = 4
    acqs = [acquisition(bolfi, BolfiOptions()) for _ in 1:acq_samples]

    # - - - PLOT - - - - -
    kwargs = (colorbar=true,)  # TODO

    if ToyProblem.x_dim() == 1

        p1 = plot(; title="true posterior", kwargs...)
        plot_posterior!(p1, ll_post; lims, label=nothing, step, parallel)
        plot_samples!(p1, X; new_datum, label=nothing)

        p2 = plot(; title="posterior mean", kwargs...)
        plot_posterior!(p2, post_mean; lims, label=nothing, step, parallel)
        plot_samples!(p2, X; new_datum, label=nothing)

        p3 = plot(; title="abs. value of GP mean", kwargs...)
        plot_posterior!(p3, x -> abs(gp_post(x)[1][1]); lims, label=nothing, step, parallel)
        plot_samples!(p3, X; new_datum, label=nothing)

        p4 = plot(; title="acquisition samples", kwargs...)
        for a in acqs
            plot_posterior!(p4, a; lims, label=nothing, step, parallel, normalize=true, grid=acq_grid)
        end
        plot_posterior!(p4, post_var; lims, label="posterior variance", step, parallel, normalize=true, linewidth=2, style=:dash, grid=acq_grid)
        plot_grid!(p4, acq_grid)
        plot_samples!(p4, X; new_datum, label="data")

        t = plot(; title, framestyle=:none, bottom_margin=-80Plots.px)
        p = plot(t, p1, p2, p3, p4; layout=@layout([°{0.05h}; [° °; ° °;]]), size=(1440, 810))
   
    elseif ToyProblem.x_dim() == 2
        
        p1 = plot(; title="posterior mean", kwargs...)
        plot_posterior!(p1, post_mean; lims, label=nothing, step, parallel)
        plot_samples!(p1, X; new_datum, label=nothing)

        p2 = plot(; title="posterior variance", kwargs...)
        plot_posterior!(p2, post_var; lims, label=nothing, step, parallel, normalize=true, grid=acq_grid)
        plot_grid!(p2, acq_grid)
        plot_samples!(p2, X; new_datum, label="data")

        p3 = plot(; title="acquisition sample #1", kwargs...)
        plot_posterior!(p3, acqs[1]; lims, label=nothing, step, parallel, normalize=true, grid=acq_grid)
        plot_grid!(p3, acq_grid)
        plot_samples!(p3, X; new_datum, label="data")

        p4 = plot(; title="acquisition sample #2", kwargs...)
        plot_posterior!(p4, acqs[2]; lims, label=nothing, step, parallel, normalize=true, grid=acq_grid)
        plot_grid!(p4, acq_grid)
        plot_samples!(p4, X; new_datum, label="data")

        p5 = plot(; title="acquisition sample #3", kwargs...)
        plot_posterior!(p5, acqs[3]; lims, label=nothing, step, parallel, normalize=true, grid=acq_grid)
        plot_grid!(p5, acq_grid)
        plot_samples!(p5, X; new_datum, label="data")

        p6 = plot(; title="acquisition sample #4", kwargs...)
        plot_posterior!(p6, acqs[4]; lims, label=nothing, step, parallel, normalize=true, grid=acq_grid)
        plot_grid!(p6, acq_grid)
        plot_samples!(p6, X; new_datum, label="data")

        t = plot(; title, framestyle=:none, bottom_margin=-80Plots.px)
        p = plot(t, p1, p2, p3, p4, p5, p6; layout=@layout([°{0.05h}; [° °; ° °; ° °;]]), size=(1440, 1215))
   
    else
        error("unsupported x_dim")
    end

    display && Plots.display(p)
    return p
end

function plot_posterior!(p, ll; lims, label=nothing, step=0.05, parallel=true, grid=nothing, normalize=false, kwargs...)
    # normalize && (ToyProblem.x_dim() > 1) && @warn "`normalize` kwarg only works for `x_dim = 1`" # TODO
    grid = isnothing(grid) ? (lims[1]:step:lims[2]) : grid

    if ToyProblem.x_dim() == 1
        # vals = ((a)->ll([a])).(grid)

        # parallel
        vals = zeros(length(grid))
        Threads.@threads for i in eachindex(vals)
            vals[i] = ll(grid[i:i])
        end

        normalize && (vals = (vals .- minimum(vals)) ./ (maximum(vals) - minimum(vals)))
        plot!(p, grid, vals; label, kwargs...)
    
    elseif ToyProblem.x_dim() == 2
        # vals = (xs->ll([xs...])).(Iterators.product(grid, grid))
    
        # parallel
        xs = Iterators.product(grid, grid) |> collect
        vals = zeros(size(xs))
        Threads.@threads for c in 1:(size(xs)[2])
            for r in 1:(size(xs)[1])
                vals[r,c] = ll([xs[r,c]...])
            end
        end

        normalize && (vals = (vals .- minimum(vals)) ./ (maximum(vals) - minimum(vals)))
        contourf!(p, grid, grid, vals'; label, kwargs...)

    else
        @error "Unsupported x dimension."
    end
    
    # # "OBSERVATION-RULES"
    # obs_color = :gold
    # plot!(p, a->y_obs[1]/a, grid; y_lims=lims, label, color=obs_color)
end

function plot_samples!(p, samples; new_datum=nothing, label=nothing)
    if ToyProblem.x_dim() == 1
        scatter!(p, vec(samples), fill(0., length(samples)); label, color=:green)
        isnothing(new_datum) || scatter!(p, new_datum, [0.]; label=nothing, color=:red)

    elseif ToyProblem.x_dim() == 2
        scatter!(p, [θ[1] for θ in eachcol(samples)], [θ[2] for θ in eachcol(samples)]; label, color=:green)
        isnothing(new_datum) || scatter!(p, [new_datum[1]], [new_datum[2]]; label=nothing, color=:red)
    
    else
        @error "Unsupported x dimension."
    end
end

function plot_grid!(p, acq_grid; label="eval. grid")
    if ToyProblem.x_dim() == 1
        scatter!(p, acq_grid, zeros(length(acq_grid)); label, color=:cyan)

    elseif ToyProblem.x_dim() == 2
        points = Iterators.product(acq_grid, acq_grid)
        xs = vec(first.(points))
        ys = vec(last.(points))
        scatter!(p, xs, ys; label, color=:cyan)
    end
end


# - - - Plot Parameter Slices - - - - -

function plot_param_slices(bolfi::BolfiProblem; options, samples=20_000, display=true, step=0.05)
    options.info && @info "Plotting ..."
    x_prior = bolfi.x_prior
    
    @assert x_prior isa Product  # individual priors must be independent
    param_samples = rand(x_prior, samples)

    return [plot_param_post(bolfi, i, param_samples; display, step) for i in 1:BOLFI.x_dim(bolfi)]
end

function plot_param_post(bolfi, param_idx, param_samples; display, step=0.05)
    bounds = bolfi.problem.domain.bounds
    x_prior = bolfi.x_prior
    param_range = bounds[1][param_idx]:step:bounds[2][param_idx]
    param_samples_ = deepcopy(param_samples)
    title = "Param $param_idx posterior"

    # true posterior
    function true_post(x)
        y = ToyProblem.experiment(x; noise_std=zeros(ToyProblem.y_dim))
        ll = pdf(MvNormal(y, ToyProblem.σe_true), ToyProblem.y_obs)
        pθ = pdf(x_prior, x)
        return pθ * ll
    end
    py = evidence(true_post, x_prior; xs=param_samples)
    function true_post_slice(xi)
        param_samples_[param_idx, :] .= xi
        return mean(true_post.(eachcol(param_samples_))) / py
    end

    # expected posterior
    exp_post = BOLFI.posterior_mean(bolfi; normalize=true)
    function exp_post_slice(xi)
        param_samples_[param_idx, :] .= xi
        return exp_post.(eachcol(param_samples_)) |> mean
    end

    # plot
    p = plot(; title)
    plot!(p, true_post_slice, param_range; label="true posterior")
    plot!(p, exp_post_slice, param_range; label="expected posterior")
    display && Plots.display(p)
    return p
end


end # module Plot
