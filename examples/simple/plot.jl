using Plots

function init_plotting(; save_plots, plot_dir)
    if save_plots
        if isdir(plot_dir)
            rm(plot_dir, recursive=true)
        end
        mkdir(plot_dir)
    end
end

function separate_new_datum(problem)
    bolfi = deepcopy(problem)
    new = bolfi.problem.data.X[:,end]
    bolfi.problem.data.X = bolfi.problem.data.X[:, 1:end-1]
    bolfi.problem.data.Y = bolfi.problem.data.Y[:, 1:end-1]
    return bolfi, new
end

function plot_state(problem; display=true, save_plots=false, plot_dir=".", plot_name="p", noise_vars_true, acquisition)
    bolfi, new_datum = separate_new_datum(problem)
    p = plot_samples(bolfi; new_datum, display, noise_vars_true, acquisition)
    save_plots && savefig(p, plot_dir * '/' * plot_name * ".png")
end

function plot_samples(bolfi; new_datum=nothing, display=true, noise_vars_true, acquisition)
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
        y = ToyProblem.experiment(x; noise_vars=zeros(ToyProblem.y_dim))
        
        # ps = numerical_issues(x) ? 0. : 1.
        isnothing(y) && return 0.

        ll = pdf(MvNormal(y, sqrt.(noise_vars_true)), ToyProblem.y_obs)
        pθ = pdf(x_prior, x)
        return pθ * ll
    end

    # gp-approximated posterior likelihood
    post_μ = BOLFI.posterior_mean(gp_post, x_prior, bolfi.var_e; normalize=false)
    ll_gp(a, b) = post_μ([a, b])

    # acquisition
    acq = acquisition(bolfi, BolfiOptions())

    # - - - PLOT - - - - -
    kwargs = (colorbar=false,)

    p1 = plot(; title="true posterior", kwargs...)
    plot_posterior!(p1, ll_post; ToyProblem.y_obs, lims, label=nothing, step)
    plot_samples!(p1, X; new_datum, label=nothing)

    p2 = plot(; title="posterior mean", kwargs...)
    plot_posterior!(p2, ll_gp; ToyProblem.y_obs, lims, label=nothing, step)
    plot_samples!(p2, X; new_datum, label=nothing)

    p3 = plot(; title="abs. value of GP mean", kwargs...)
    plot_posterior!(p3, (a,b) -> abs(gp_post([a,b])[1][1]); ToyProblem.y_obs, lims, label=nothing, step)
    plot_samples!(p3, X; new_datum, label=nothing)

    p4 = plot(; title="posterior variance", kwargs...)
    plot_posterior!(p4, (a,b) -> acq([a,b]); ToyProblem.y_obs, lims, label=nothing, step)
    plot_samples!(p4, X; new_datum, label=nothing)

    title = ""
    t = plot(; title, framestyle=:none, bottom_margin=-80Plots.px)
    p = plot(t, p1, p2, p3, p4; layout=@layout([°{0.05h}; [° °; ° °;]]), size=(1440, 810))
    display && Plots.display(p)
    return p
end

function plot_posterior!(p, ll; y_obs, lims, label="ab=d", step=0.05)
    grid = lims[1]:step:lims[2]
    contourf!(p, grid, grid, ll)
    
    # "OBSERVATION-RULES"
    obs_color = :gold
    plot!(p, a->y_obs[1]/a, grid; y_lims=lims, label, color=obs_color)
end

function plot_samples!(p, samples; new_datum=nothing, label="(a,b) ~ p(a,b|d)")
    scatter!(p, [θ[1] for θ in eachcol(samples)], [θ[2] for θ in eachcol(samples)]; label, color=:green)
    isnothing(new_datum) || scatter!(p, [new_datum[1]], [new_datum[2]]; label=nothing, color=:red)
end
