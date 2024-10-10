using BOLFI
using BOSS
using Distributions
using OptimizationPRIMA

using Random
Random.seed!(555)

include("toy_problem.jl")
# include("plot.jl")
include("makie/makie.jl")

# TODO
include("boss_override.jl")

# MAIN SCRIPT
function main(;
    init_data=3, # TODO
)
    # TODO
    # problem = ToyProblem.bolfi_problem(init_data)
    X = hcat(zeros(ToyProblem.x_dim())) # TODO
    problem = ToyProblem.bolfi_problem(ExperimentDataPrior(X, reduce(hcat, ToyProblem.obj.(eachcol(X)))))

    parallel = true

    # TODO
    gridstep = 0.5 # TODO
    x_grid = -5.:gridstep:5.

    # model_fitter = SamplingMAP(;
    #     samples = 200, # TODO
    #     parallel,
    # )
    model_fitter = OptimizationMAP(;
        algorithm = NEWUOA(),
        parallel,
        multistart = 200, # TODO
        rhoend = 1e-2, # TODO
    )

    grid_am_ = GridAM(;
        problem.problem,
        steps = fill(gridstep, ToyProblem.x_dim()),
        parallel,
    )
    acq_maximizer = grid_am_
    # acq_maximizer = OptimizationAM(;
    #     algorithm = BOBYQA(),
    #     parallel,
    #     multistart = 200, # TODO
    #     rhoend = 1e-2, # TODO
    # )

    # acquisition = PostVarAcq()
    acquisition = InfoGain(;
        samples = 100, # TODO 1000
        θ_grid = reduce(hcat, grid_am_.points),
        δ_kernel = BOSS.GaussianKernel(),
        p_kernel = BOSS.GaussianKernel(),
    )

    term_cond = BOSS.IterLimit(10) # TODO
    # term_cond = AEConfidence(;
    #     max_iters = 50,
    #     samples = 10_000,
    #     q = 0.95,
    #     r = 0.8,  # TODO
    # )

    plt = MakiePlots.PlotCallback(;
        title = "",
        plot_each = 1,  # TODO
        display = true,
        save_plots = true,
        plot_dir = "./examples/simple/plots/_new_",
        plot_name = "p",
        noise_std_true = ToyProblem.σe_true,
        step = 0.05,  # TODO
        parallel,
        acq_grid = x_grid,
    )
    options = BolfiOptions(;
        callback = plt,  # TODO
        # callback = NoCallback(),
    )

    # # TODO remove
    # BOSS.estimate_parameters!(problem.problem, model_fitter)
    # acq = acquisition(problem, options)
    # return problem, acq

    # TODO rem
    @show ToyProblem.x_dim()

    MakiePlots.init_plotting(plt)
    bolfi!(problem; acquisition, model_fitter, acq_maximizer, term_cond, options)
    # Plot.plot_final(plt; acquisition, model_fitter, acq_maximizer, term_cond, options)
    # Plot.plot_param_slices(problem; options, samples=2_000, step=0.05)
    
    return problem
    # MakiePlots.plot_lengthscales(problem, acquisition, x_grid)
end
