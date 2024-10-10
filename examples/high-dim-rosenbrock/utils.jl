
data_dir(NAME, ID) = "./examples/high-dim-rosenbrock/data/$NAME-$ID"
data_dir(NAME, ::Nothing) = "./examples/high-dim-rosenbrock/data/$NAME"

"""
Callback which measures the elpased time of during each iteration.

Use `times(::Stopwatch)` to get the iteration times in seconds.
"""
struct Stopwatch <: BolfiCallback
    timestamps::Vector{Float64}
end
Stopwatch() = Stopwatch(Float64[])

function (cb::Stopwatch)(p::BolfiProblem; kwargs...)
    push!(cb.timestamps, time())
end

function times(cb::Stopwatch)
    return cb.timestamps[2:end] .- cb.timestamps[1:end-1]
end
