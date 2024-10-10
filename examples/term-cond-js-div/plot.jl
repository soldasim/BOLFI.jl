module Plot

using Plots
using Distributions
using BOSS, BOLFI
using OptimizationPRIMA

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
    noise_std_true::Vector{Float64}
end
PlotCallback(;
    plot_each::Int = 1,
    display::Bool = true,
    save_plots::Bool = false,
    plot_dir::String = ".",
    plot_name::String = "p",
    noise_std_true::Vector{Float64},
) = PlotCallback(nothing, 0, plot_each, display, save_plots, plot_dir, plot_name, noise_std_true)

"""
Plots the state in the current iteration.
"""
function (plt::PlotCallback)(bolfi::BolfiProblem; acquisition, options, term_cond, first, kwargs...)
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
        plot_state(plt.prev_state, new_datum, term_cond.term_cond; display=plt.display, save_plots=plt.save_plots, plot_dir=plt.plot_dir, plot_name=plt.plot_name*"_$plot_iter", noise_std_true=plt.noise_std_true, acquisition=acquisition.acq)
    end
    
    plt.prev_state = deepcopy(bolfi)
    plt.iters += 1
end

"""
Plot the final state after the BO procedure concludes.
"""
function plot_final(plt::PlotCallback; acquisition, options, term_cond, kwargs...)
    plot_iter = plt.iters - 1
    options.info && @info "Plotting ..."
    plot_state(plt.prev_state, nothing, term_cond; display=plt.display, save_plots=plt.save_plots, plot_dir=plt.plot_dir, plot_name=plt.plot_name*"_$plot_iter", noise_std_true=plt.noise_std_true, acquisition)
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

function plot_state(bolfi, new_datum, term_cond; display=true, save_plots=false, plot_dir=".", plot_name="p", noise_std_true, acquisition)
    # Plots with hyperparams fitted using *all* data! (Does not really matter.)
    p = plot_samples(bolfi, term_cond; new_datum, display, noise_std_true, acquisition)
    save_plots && savefig(p, plot_dir * '/' * plot_name * ".png")
end

function plot_samples(bolfi, term_cond; new_datum=nothing, display=true, noise_std_true, acquisition)
    problem = bolfi.problem
    gp_post = BOSS.model_posterior(problem)

    x_prior = bolfi.x_prior
    bounds = problem.domain.bounds
    @assert all((lb == bounds[1][1] for lb in bounds[1]))
    @assert all((ub == bounds[2][1] for ub in bounds[2]))
    lims = bounds[1][1], bounds[2][1]
    step = 0.05
    X, Y = problem.data.X, problem.data.Y

    # unnormalized posterior likelihood `p(d | a, b) * p(a, b) ∝ p(a, b | d)`
    function ll_post(a, b)
        x = [a, b]
        y = ToyProblem.experiment(x; noise_std=zeros(ToyProblem.y_dim))

        ll = pdf(MvNormal(y, noise_std_true), ToyProblem.y_obs)
        pθ = pdf(x_prior, x)
        return pθ * ll
    end

    # posteriors
    post_expect = posterior_mean(gp_post, x_prior, bolfi.std_obs)
    post_approx = approx_posterior(gp_post, x_prior, bolfi.std_obs)
    # eps = 1.
    # post_lb = approx_posterior(gp_bound(gp_post, -eps), x_prior, bolfi.std_obs)
    # post_ub = approx_posterior(gp_bound(gp_post, +eps), x_prior, bolfi.std_obs)

    # acquisition
    acq = acquisition(bolfi, BolfiOptions())

    # - - - PLOT - - - - -
    kwargs = (colorbar=false,)

    # TODO hyperparams
    gauss_opt = BOLFI.GaussMixOptions(;
        algorithm = BOBYQA(),
        multistart = 200,
        parallel = true,
        rhoend = 1e-4,
        cluster_ϵ = 1.,
    )
    gauss_mix_expect = BOLFI.approx_by_gauss_mix(post_expect, problem.domain, term_cond.gauss_opt)
    gauss_mix_approx = BOLFI.approx_by_gauss_mix(post_approx, problem.domain, term_cond.gauss_opt)
    # gauss_mix_lb = BOLFI.approx_by_gauss_mix(post_lb, problem.domain, term_cond.gauss_opt)
    # gauss_mix_ub = BOLFI.approx_by_gauss_mix(post_ub, problem.domain, term_cond.gauss_opt)

    # divergence
    js_div = BOLFI.jensen_shannon_divergence(post_expect, post_approx, gauss_mix_expect, gauss_mix_approx; term_cond.samples)
    # js_div = BOLFI.jensen_shannon_divergence(post_lb, post_ub, gauss_mix_lb, gauss_mix_ub; term_cond.samples)
    js_div = round(js_div; digits=6)

    p1 = plot(; title="true posterior", kwargs...)
    plot_posterior!(p1, ll_post; ToyProblem.y_obs, lims, step)
    plot_samples!(p1, X; new_datum)

    p2 = plot(; title="expected posterior", kwargs...)
    plot_posterior!(p2, (a,b)->post_expect([a,b]); ToyProblem.y_obs, lims, step)
    plot_samples!(p2, X; new_datum)
    plot_modes!(p2, gauss_mix_expect)

    p3 = plot(; title="approx. posterior", kwargs...)
    plot_posterior!(p3, (a,b)->post_approx([a,b]); ToyProblem.y_obs, lims, step)
    plot_samples!(p3, X; new_datum)
    plot_modes!(p3, gauss_mix_approx)

    # p4 = plot(; title="LB posterior", kwargs...)
    # plot_posterior!(p4, (a,b)->post_lb([a,b]); ToyProblem.y_obs, lims, step)
    # plot_samples!(p4, X; new_datum)
    # plot_modes!(p4, gauss_mix_lb)

    # p5 = plot(; title="UB posterior", kwargs...)
    # plot_posterior!(p5, (a,b)->post_ub([a,b]); ToyProblem.y_obs, lims, step)
    # plot_samples!(p5, X; new_datum)
    # plot_modes!(p5, gauss_mix_ub)

    # p6 = plot(; title="abs. value of GP mean", kwargs...)
    # plot_posterior!(p6, (a,b) -> abs(gp_post([a,b])[1][1]); ToyProblem.y_obs, lims, step)
    # plot_samples!(p6, X; new_datum)

    p7 = plot(; title="posterior variance", kwargs...)
    plot_posterior!(p7, (a,b) -> acq([a,b]); ToyProblem.y_obs, lims, step)
    plot_samples!(p7, X; new_datum)

    p8 = plot(; title="expected Gauss-mix", kwargs...)
    plot_gauss_mix!(p8, gauss_mix_expect; ToyProblem.y_obs, lims, step)
    plot_samples!(p8, X; new_datum)
    plot_modes!(p8, gauss_mix_expect)
    annotate!(p8, 0., -4.5, ("JS-div: $js_div", :cyan))

    p9 = plot(; title="approx. Gauss-mix", kwargs...)
    plot_gauss_mix!(p9, gauss_mix_approx; ToyProblem.y_obs, lims, step)
    plot_samples!(p9, X; new_datum)
    plot_modes!(p9, gauss_mix_approx)
    annotate!(p9, 0., -4.5, ("JS-div: $js_div", :cyan))

    # p10 = plot(; title="LB Gauss-mix", kwargs...)
    # plot_gauss_mix!(p10, gauss_mix_lb; ToyProblem.y_obs, lims, step)
    # plot_samples!(p10, X; new_datum)
    # plot_modes!(p10, gauss_mix_lb)
    # annotate!(p10, 0., -4.5, ("JS-div: $js_div", :cyan))

    # p11 = plot(; title="UB Gauss-mix", kwargs...)
    # plot_gauss_mix!(p11, gauss_mix_ub; ToyProblem.y_obs, lims, step)
    # plot_samples!(p11, X; new_datum)
    # plot_modes!(p11, gauss_mix_ub)
    # annotate!(p11, 0., -4.5, ("JS-div: $js_div", :cyan))

    title = ""
    t = plot(; title, framestyle=:none, bottom_margin=-80Plots.px)
    # p = plot(t, p1, p2, p4, p5; layout=@layout([°{0.05h}; [° °; ° °;]]), size=(1440, 810))
    p = plot(t, p1, p7, p3, p2, p9, p8; layout=@layout([°{0.05h}; [° °; ° °; ° °;]]), size=(1440, 1215))
    # p = plot(t, p1, p7, p4, p5, p10, p11; layout=@layout([°{0.05h}; [° °; ° °; ° °;]]), size=(1440, 1215))
    display && Plots.display(p)
    return p
end

function plot_posterior!(p, ll; y_obs, lims, step=0.05)
    grid = lims[1]:step:lims[2]
    contourf!(p, grid, grid, ll)
    plot_observation_rules!(p; y_obs, lims, step)
end

function plot_observation_rules!(p; y_obs, lims, step=0.05)
    grid = lims[1]:step:lims[2]

    obs_color = :gold
    plot!(p, a->y_obs[1]/a, grid; y_lims=lims, label=nothing, color=obs_color)
end

function plot_samples!(p, samples; new_datum=nothing)
    scatter!(p, [θ[1] for θ in eachcol(samples)], [θ[2] for θ in eachcol(samples)]; label=nothing, color=:green)
    isnothing(new_datum) || scatter!(p, [new_datum[1]], [new_datum[2]]; label=nothing, color=:red)
end

function plot_gauss_mix!(p, gauss_mix::BOLFI.GaussMix; y_obs, lims, step=0.05)
    grid = lims[1]:step:lims[2]
    contourf!(p, grid, grid, (a,b) -> pdf(gauss_mix, [a,b]))
    plot_observation_rules!(p; y_obs, lims, step)
end

function plot_modes!(p, gauss_mix::BOLFI.GaussMix)
    modes = hcat(getfield.(gauss_mix.components, :μ)...)
    scatter!(p, modes[1,:], modes[2,:]; label="$(size(modes)[2]) modes", color=:cyan)
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
