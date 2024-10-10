
include("main.jl")

function quality_script()
    probs = BolfiProblem[]
    opts = BolfiOptions[]

    plt = plot(; title="approximation quality")

    for x_dim in 1:5
        # redefine problem
        @eval ToyProblem.x_dim() = x_dim

        # run BOLFI
        p, opt = script_bolfi()
        push!(probs, p)
        push!(opts, opt)

        # plot
        PlotQuality.plot_approx_quality(p; options=opt, p=plt, label="x_dim = $x_dim")
        savefig(plt, "examples/simple/plots" * '/' * "quality_$x_dim" * ".png")
    end

    return plt, probs, opts
end
