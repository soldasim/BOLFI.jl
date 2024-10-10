using BOLFI
using BOSS
using Distributions
using OptimizationPRIMA
using Serialization

using Random
Random.seed!(555)

include("toy_problem.jl")
include("plot.jl")

# MAIN SCRIPT
function script_bolfi(;
    init_data=3,
)
    problem = ToyProblem.bolfi_problem(init_data)
    domain = problem.problem.domain
    acquisition = PostVarAcq()

    # model_fitter = SamplingMAP(;
    #     samples = 200,
    #     parallel = true,
    # )
    model_fitter = OptimizationMAP(
        algorithm = NEWUOA(),
        parallel = true,
        multistart = 200,  # TODO
        # rhobeg = 1e-1,  # TODO
        rhoend = 1e-4,  # TODO
    )
    # acq_maximizer = GridAM(;
    #     problem.problem,
    #     steps = fill(0.05, ToyProblem.x_dim()),
    #     parallel = true,
    # )
    acq_maximizer = OptimizationAM(;
        algorithm = BOBYQA(),
        parallel = true,
        multistart = 200,  # TODO
        # rhobeg = 1e-1,  # TODO
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
        target_mmd = 0., # TODO
    )

    plt = Plot.PlotCallback(;
        plot_each = 1, # TODO
        display = true,
        save_plots = true,
        plot_dir = "./examples/term-cond-opt-transport/plots/__new__",
        plot_name = "p",
        noise_std_true = ToyProblem.σe_true,
    )
    options = BolfiOptions(;
        callback = plt,
    )

    Plot.init_plotting(plt)
    bolfi!(problem; acquisition, model_fitter, acq_maximizer, term_cond, options)
    Plot.plot_final(plt; acquisition, model_fitter, acq_maximizer, term_cond, options)
    
    # save data
    serialize(plt.plot_dir * '/' * "problem", problem)
    serialize(plt.plot_dir * '/' * "data", problem.problem.data)
    serialize(plt.plot_dir * '/' * "term_cond", term_cond)
    serialize(plt.plot_dir * '/' * "history", term_cond.history)

    Plot.plot_param_slices(problem; options, samples=2_000, step=0.05)
    return problem
end
