using Distributions
using Random

include("toy_problem.jl")

function test_cost_std()
    @info "Resetting seed!"
    Random.seed!(555)

    init_data = 3
    p = ToyProblem.bolfi_problem(init_data)

    model_fitter = OptimizationMAP(
        algorithm = NEWUOA(),
        parallel = true,
        multistart = 200,  # TODO
        # rhobeg = 1e-1,  # TODO
        rhoend = 1e-4,  # TODO
    )

    BOSS.estimate_parameters!(p.problem, model_fitter)

    term_cond = OptTransport(;
        max_iters = 50,
        samples = 2000,
        target_cost = 0.5,
        gauss_opt = GaussMixOptions(;
            algorithm = BOBYQA(),
            multistart = 200,
            parallel = true,
            cluster_ϵ = nothing,
            rel_min_weight = 1e-8,
            rhoend = 1e-4,
        ),
        # metric = BOLFI.SqEuclidean(),
        epsilon = 1.,
    )

    costs = [BOLFI.calculate(term_cond, p) for _ in 1:20]
    @show std(costs)
    return costs
end
