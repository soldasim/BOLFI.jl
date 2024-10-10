using Serialization

function save_problem(dir, p::BolfiProblem)
    serialize(dir * '/' * "bolfi", p)
    serialize(dir * '/' * "boss", p.problem)
    serialize(dir * '/' * "data", deconstruct(p.problem.data))
end

function save_times(dir, times::Vector{Float64})
    serialize(dir * '/' * "times", times)
end

load(file) = deserialize(file)

function deconstruct(data::ExperimentDataPrior)
    return (
        data.X,
        data.Y,
    )
end
function deconstruct(data::ExperimentDataPost)
    return (
        data.X,
        data.Y,
        data.params,
        data.consistent
    )
end
