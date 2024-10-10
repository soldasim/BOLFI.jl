module ToyProblem

using BOLFI
using BOSS
using Distributions
using Random


# - - - ROSENBROCK FUNCTION - - - - -

const B = 1.
OBS_POINT_PRIOR(dim::Int) = Product(fill(Uniform(-5., 5.), dim))

# `input` are the observation locations and determine `y_dim`
# `params` are the parameters of interest and deterimne `x_dim`
# `length(params) + 1 == length(input)`
rosenbrock(input, params) = mean((params .* (input[2:end] .- (input[1:end-1] .^ 2)) .^ 2) .+ ((B .- input[1:end-1]) .^ 2))

struct RosenbrockExperiment
    observation_points::Matrix{Float64}
end
function RosenbrockExperiment(x_dim::Int, y_dim::Int; seed=555)
    input_dim = x_dim + 1
    output_dim = y_dim
    
    @info "Generating Rosenbrock observation points with seed $seed."
    Random.seed!(seed)
    obs_points = rand(OBS_POINT_PRIOR(input_dim), output_dim)

    return RosenbrockExperiment(obs_points)
end

function (exp::RosenbrockExperiment)(x::AbstractVector{<:Real})
    return rosenbrock.(eachcol(exp.observation_points), Ref(x))
end


# - - - SETTINGS - - - - -

struct ToySettings
    x_dim::Int
    y_dim::Int
    f::Any
    x_true::Vector{Float64}     # the true parameter values
    y_obs::Vector{Float64}      # the observed values
    rel_err::Float64            # relative error of the observation
    std_obs::Vector{Float64}    # the std of the observation
    std_sim::Vector{Float64}    # the std of the simulation noise
end
function ToySettings(;
    x_dim = 1,
    y_dim = 1,
    rel_err = 0.05,
    std_sim = 0.01,
    x_true = 0.8,
)
    f = normalized(RosenbrockExperiment(x_dim, y_dim))
    x_true = fill(x_true, x_dim)
    y_obs = f(x_true)
    std_obs = rel_err * y_obs
    std_sim = fill(std_sim, y_dim)
    return ToySettings(x_dim, y_dim, f, x_true, y_obs, rel_err, std_obs, std_sim)
end

# normalize inputs and outputs to reasonable values
const x_norm = 100.
const y_norm = 100.

function normalized(f)
    function nf(x)
        x *= x_norm
        y = f(x)
        y /= y_norm
        return y
    end
end


# - - - OBJECTIVE & SIMULATION - - - - -

# The objective queried by the algorithm
function objective(s::ToySettings)
    sim = simulation(s)
    return x -> sim(x) .- s.y_obs
end

# The "simulation". (approximates the "experiment")
function simulation(s::ToySettings)
    noise_dist = MvNormal(zeros(s.y_dim), s.std_sim)
    return x -> s.f(x) + rand(noise_dist)
end


# - - - HYPERPARAMETERS - - - - -

get_bounds(s::ToySettings) = (fill(0.5, s.x_dim), fill(1.5, s.x_dim))

# get_x_prior(s::ToySettings) = Product(Uniform.(get_bounds(s)...))
get_x_prior(s::ToySettings) = Product(fill(Normal(1., 0.5), s.x_dim))

get_kernel(s::ToySettings) = BOSS.Matern32Kernel()

const λ_MIN = 0.01
const λ_MAX = 1.
get_length_scale_priors(s::ToySettings) = fill(Product(fill(calc_inverse_gamma(λ_MIN, λ_MAX), s.x_dim)), s.y_dim)

const α_MIN = 0.1
const α_MAX = 10.
get_amplitude_priors(s::ToySettings) = fill(calc_inverse_gamma(α_MIN, α_MAX), s.y_dim)

function get_noise_std_priors(s::ToySettings)
    μ_std = s.std_sim
    max_std = 10 * s.std_sim
    return [truncated(Normal(μ_std[i], max_std[i] / 3); lower=0.) for i in 1:s.y_dim]
end


# - - - INITIALIZATION - - - - -

function get_init_data(s::ToySettings, count::Int)
    obj = objective(s)
    X = reduce(hcat, (random_datapoint(s) for _ in 1:count))[:,:]
    Y = reduce(hcat, (obj(x) for x in eachcol(X)))[:,:]
    return BOSS.ExperimentDataPrior(X, Y)
end
function get_init_data(s::ToySettings, X::Matrix{Float64})
    obj = objective(s)
    Y = reduce(hcat, (obj(x) for x in eachcol(X)))[:,:]
    return BOSS.ExperimentDataPrior(X, Y)
end

bolfi_problem(s::ToySettings, init_data) = bolfi_problem(s, get_init_data(s, init_data))

function bolfi_problem(s::ToySettings, data::ExperimentData)
    return BolfiProblem(data;
        f = objective(s),
        bounds = get_bounds(s),
        kernel = get_kernel(s),
        length_scale_priors = get_length_scale_priors(s),
        amp_priors = get_amplitude_priors(s),
        noise_std_priors = get_noise_std_priors(s),
        std_obs = s.std_obs,
        x_prior = get_x_prior(s),
    )
end


# - - - UTILS - - - - -

function random_datapoint(s::ToySettings)
    x_prior = get_x_prior(s)
    bounds = get_bounds(s)

    x = rand(x_prior)
    while !BOSS.in_bounds(x, bounds)
        x = rand(x_prior)
    end
    return x
end

"""
Return an _approximate_ Inverse Gamma distribution
with 0.99 probability mass between `lb` and `ub.`
"""
function calc_inverse_gamma(lb, ub)
    μ = (ub + lb) / 2
    σ = (ub - lb) / 6
    a = (μ^2 / σ^2) + 2
    b = μ * ((μ^2 / σ^2) + 1)
    return InverseGamma(a, b)
end

end # module ToyProblem
