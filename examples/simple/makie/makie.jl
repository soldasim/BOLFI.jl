module MakiePlots

using CairoMakie

using Distributions
using BOSS, BOLFI

include("../toy_problem.jl")


# --- CALLBACK ---

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
        plot_state(plt.prev_state, new_datum; plt, acquisition, iter=plot_iter)
    end
    
    plt.prev_state = deepcopy(bolfi)
    plt.iters += 1
end


# --- INIT ---

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


# --- PLOTS ---

function plot_state(prev_state::BolfiProblem, new_datum::AbstractVector{<:Real}; plt::PlotCallback, iter::Int, kwargs...)
    if ToyProblem.x_dim() == 1
        f = plot_state_1d(prev_state, new_datum; plt, kwargs...)
    elseif ToyProblem.x_dim() == 2
        f = plot_state_2d(prev_state, new_datum; plt, kwargs...)
    else
        error("unsupported dimension")
    end

    plt.display && display(f)
    plt.save_plots && save(plt.plot_dir * '/' * plt.plot_name * "_$iter" * ".png", f)
    return f
end

include("utils.jl")
include("1d.jl")
include("2d.jl")
include("hyperparams.jl")

end # MakiePlots