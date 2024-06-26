using BOLFI
using BOSS
using Distributions

using Random
Random.seed!(555)

include("toy_problem.jl")
include("plot.jl")

# MAIN SCRIPT
# Change the experiment by changing the `const mode` in `toy_problem.jl`.
function script_bolfi(;
    init_data=3,
)
    problem = ToyProblem.bolfi_problem(init_data)
    
    if ToyProblem.mode == Val(:T1)
        acquisition = PostVariance()
    else
        acquisition = SetsPostVariance()
    end

    model_fitter = BOSS.SamplingMLE(;
        samples = 200,
        parallel = true,
    )
    acq_maximizer = BOSS.GridAM(;
        problem.problem,
        steps = [0.05, 0.05],
        parallel = true,
    )

    # EXPERIMENTAL TERMINATION CONDITIONS
    xs = rand(problem.x_prior, 10_000)
    max_iters = 50  # TODO

    term_cond = AEConfidence(;
        xs,
        q = 0.8,
        r = 0.95,
        max_iters,
    )
    # term_cond = UBLBConfidence(;
    #     xs = rand(problem.x_prior, samples),
    #     n = 1.,
    #     q = 0.8,
    #     r = 0.8,
    #     max_iters
    # )
    # term_cond = IterLimit(25);

    save_plots = true
    plot_dir = "./examples/diagnostic_selection/plots"

    options = BolfiOptions(;
        callback = Plot.PlotCallback(;
            plot_each = 5,  # TODO
            term_cond,
            save_plots,
            plot_dir,
            put_in_scale = false,
        ),
    )

    Plot.init_plotting(; save_plots, plot_dir)
    bolfi!(problem; acquisition, model_fitter, acq_maximizer, term_cond, options)

    # final state  # TODO
    Plot.plot_state(problem; term_cond, iter=options.callback.iters, save_plots, plot_dir, plot_name="p_final", acquisition)
    
    return problem
end
