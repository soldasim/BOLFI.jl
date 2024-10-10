
const ID = ARGS[1]
const NAME = ARGS[2]
const X_DIM = ARGS[3]
const Y_DIM = ARGS[4]

using Pkg
Pkg.activate("/home/soldasim/BOLFI.jl/examples")
# Pkg.resolve()
# Pkg.instantiate()
include("/home/soldasim/BOLFI.jl/examples/high-dim/main.jl")

@show NAME
@show ID

x_dim = parse(Int, X_DIM)
y_dim = parse(Int, Y_DIM)
t = @elapsed main(; NAME, ID, x_dim, y_dim)

println()
@show t
