using BOLFI
using BOSS
using Distributions
using OptimizationPRIMA

include("toy_problem.jl")
include("serialization.jl")
include("utils.jl")

function test_grid()
    for x_dim in 2:2:10
        for y_dim in 2:2:10
            main(; NAME="cost", ID="$x_dim-$y_dim", x_dim, y_dim)
        end
    end
end

function test_termcond(;
    NAME,
    ID,
    x_dim = 1,
    y_dim = 1,
    init_data = 10,    
)
    s = ToyProblem.ToySettings(; x_dim, y_dim)
    # init_data = ToyProblem.get_bounds(s) |> mean |> hcat
    problem = ToyProblem.bolfi_problem(s, init_data)
    domain = problem.problem.domain

    model_fitter = SampleOptMAP(
        samples = 200_000,  # TODO
        algorithm = NEWUOA(),
        parallel = true,
        multistart = 200,  # TODO
        rhobeg = 1e-1,  # TODO
        rhoend = 1e-4,  # TODO
    )

    BOSS.estimate_parameters!(problem.problem, model_fitter)

    term_cond = term_cond = MaximumMeanDiscrepancy(;
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

    times = Float64[]
    vals = Union{Nothing, Float64}[]
    for _ in 1:20 # TODO
        t_ = @elapsed val_ = BOLFI.calculate(term_cond, problem)
        push!(times, t_)
        push!(vals, val_)
    end

    isdir(data_dir(NAME, ID)) || mkdir(data_dir(NAME, ID))
    serialize(data_dir(NAME, ID) * '/' * "times", times)
    serialize(data_dir(NAME, ID) * '/' * "vals", vals)
end
