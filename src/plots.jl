
"""
    PlotSettings(; kwargs...)

Aggregates all plot settings for the `plot_marginals_int` and `plot_marginals_kde` functions.

# Kwargs
- `plot_step::Float64`: Controls the plot resolution. Greatly impacts the computational cost of plotting.
- `param_labels::Union{Nothing, Vector{String}}`: Labels used for the parameters in the plots.
        Defaults to `nothing`, in which case the default labels "x1,x2,..." are used.
"""
abstract type PlotSettings end

"""
    using CairoMakie
    plot_marginals_int(::BolfiProblem)

Create a matrix of plots displaying the marginal posterior distribution of each pair of parameters
with the individual marginals of each parameter on the diagonal.

Approximates the marginals by numerically integrating the marginal integrals
over a generated latin hypercube grid of parameter samples.

# Kwargs

- `grid_size::Int`: The number of samples in the generate LHC grid.
        The higher the number, the more precise marginal plots.
- `plot_settings::PlotSettings`: Settings for the plotting.
- `info::Bool`: Set to `false` to disable prints.
- `display::Bool`: Set to `false` to not display the figure. It is still returned.
"""
function plot_marginals_int end

"""
    using CairoMakie, Turing
    plot_marginals_kde(::BolfiProblem)

Create a matrix of plots displaying the marginal posterior distribution of each pair of parameters
with the individual marginals of each parameter on the diagonal.

Approximates the marginals by kernel density estimation
over parameter samples drawn by MCMC methods from the Turing.jl package.

One should experiment with different kernel length scales to obtain a good approximation
of the marginals. In case of a `Kernel` from `KernelFunctions`, the length scale can be changed
via the `with_lengthscale` function. (E.g. `with_lengthscale(GaussianKernel(), 0.1)`)

# Kwargs

- `turing_options::TuringOptions`: Settings for the MCMC sampling.
- `kernel::Kernel`: The kernel used in the KDE. Either as an instance of `KernelFunctions.Kernel`
        or any function with signature `(x::AbstractVector{<:Real}, x_::AbstractVector{<:Real}) -> ::Real`.
- `plot_settings::PlotSettings`: Settings for the plotting.
- `info::Bool`: Set to `false` to disable prints.
- `display::Bool`: Set to `false` to not display the figure. It is still returned.
"""
function plot_marginals_kde end

# The plotting is implemented in the `CairoExt` extension.
