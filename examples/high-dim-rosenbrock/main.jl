using BOLFI
using BOSS
using Distributions
using OptimizationPRIMA
using Plots

include("toy_problem.jl")
include("plot.jl")
include("serialization.jl")
include("utils.jl")

function main(;
    NAME = "__new__",
    ID = nothing,
    x_dim = 1,
    y_dim = 1,
    init_data = 10,
)
    s = ToyProblem.ToySettings(; x_dim, y_dim)
    # init_data = ToyProblem.get_bounds(s) |> mean |> hcat
    problem = ToyProblem.bolfi_problem(s, init_data)
    domain = problem.problem.domain
    
    acquisition = PostVarAcq()

    # model_fitter = SamplingMAP(;
    #     samples = 2000,  # TODO
    #     parallel = true,
    # )
    # model_fitter = OptimizationMAP(
    #     algorithm = NEWUOA(),
    #     parallel = false,
    #     multistart = 200,  # TODO
    #     # rhobeg = 1e-1,  # TODO
    #     rhoend = 1e-4,  # TODO
    # )
    model_fitter = SampleOptMAP(
        samples = 200_000,  # TODO
        algorithm = NEWUOA(),
        parallel = true,
        multistart = 200,  # TODO
        rhobeg = 1e-1,  # TODO
        rhoend = 1e-4,  # TODO
    )

    # acq_maximizer = SamplingAM(;
    #     x_prior = problem.x_prior,
    #     samples = 2000,  # TODO
    # )
    # acq_maximizer = OptimizationAM(;
    #     algorithm = BOBYQA(),
    #     parallel = false,
    #     multistart = 200,  # TODO
    #     # rhobeg = 1e-1,  # TODO
    #     rhoend = 1e-4,  # TODO
    # )
    acq_maximizer = SampleOptAM(;
        x_prior = problem.x_prior,
        samples = 200_000,  # TODO
        max_attempts = 10,
        algorithm = BOBYQA(),
        parallel = true,
        multistart = 200,  # TODO
        rhobeg = 1e-1,  # TODO
        rhoend = 1e-4,  # TODO
    )

    # TODO
    term_cond = MaximumMeanDiscrepancy(;
        max_iters = 50, # TODO
        samples = 10_000, # TODO
        gauss_opt = GaussMixOptions(;
            algorithm = BOBYQA(),
            multistart = 200,
            parallel = true,
            cluster_ϵ = nothing,
            rel_min_weight = 1e-8,
            rhoend = 1e-4,
        ),
        kernel = BOLFI.GaussianKernel(),
        target_mmd = 0.01, # TODO
    )

    plt = Plot.PlotCallback(;
        plot_each = 1,  # TODO
        display = true,
        save_plots = true,
        plot_dir = data_dir(NAME, ID),
        plot_name = "p",
        std_obs = s.std_obs,
        std_sim = s.std_sim,
        y_obs = s.y_obs,
        f = s.f,
        step = 0.005,
        acquisition,
    )
    options = BolfiOptions(;
        # callback = plt,
        callback = Stopwatch(),
        # callback = NoCallback(),
    )

    # TODO rem
    @show s.x_dim
    @show s.y_dim

    # --- PRE-RUN ---
    Plot.init_plotting(plt)
    
    # --- BOLFI ---
    bolfi!(problem; acquisition, model_fitter, acq_maximizer, term_cond, options)
    
    # --- POST-RUN ---
    # Plot.plot_final(plt; acquisition, model_fitter, acq_maximizer, term_cond, options)
    plots = Plot.plot_param_slices(problem; plt, options, samples=2_000, step=0.05)
    for i in eachindex(plots)
        savefig(plots[i], plt.plot_dir * '/' * "param_$i" * ".png")
    end
    save_problem(data_dir(NAME, ID), problem)
    save_times(data_dir(NAME, ID), times(options.callback))
    serialize(data_dir(NAME, ID) * '/' * "history", term_cond.history)

    return problem
end
