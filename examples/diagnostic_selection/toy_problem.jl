module ToyProblem

using BOLFI
using BOSS
using Distributions

# - - - PROBLEM - - - - -

# CHANGE THE EXPERIMENT HERE
const mode = Val(:T1)
# const mode = Val(:T2)

"""observation"""
y_obs(::Val{:T1}) = [1.]
y_dim(::Val{:T1}) = 1
y_obs(::Val{:T2}) = [1., 0.]
y_dim(::Val{:T2}) = 2

"""observation noise std"""
σe_true(::Val{:T1}) = [0.5]  # true noise
σe(::Val{:T1}) =      [0.5]  # hyperparameter
σe_true(::Val{:T2}) = [0.5, 0.5]  # true noise
σe(::Val{:T2}) =      [0.5, 0.5]  # hyperparameter
"""simulation noise std"""
const ω = [0.01 for _ in 1:y_dim(mode)]

# objective functions
f_(x) = x[1] * x[2]
g_(x) = (x[2] - x[1])

get_y_sets(::Val{:T1}) = nothing
get_y_sets(::Val{:T2}) = [true;false;; false;true;;]

# The "real experiment". (for plotting only)
function experiment(m::Val{:T1}, x; noise_vars=σe_true(m).^2)
    y = [f_(x)] + rand(MvNormal(zeros(y_dim(m)), sqrt.(noise_vars)))
    return y
end
function experiment(m::Val{:T2}, x; noise_vars=σe_true(m).^2)
    y = [f_(x), g_(x)] + rand(MvNormal(zeros(y_dim(m)), sqrt.(noise_vars)))
    return y
end

# The "simulation". (approximates the "experiment")
function simulation(m::Val{:T1}, x; noise_vars=ω.^2)
    y = [f_(x)] + rand(MvNormal(zeros(y_dim(m)), sqrt.(noise_vars)))
    return y
end
function simulation(m::Val{:T2}, x; noise_vars=ω.^2)
    y = [f_(x), g_(x)] + rand(MvNormal(zeros(y_dim(m)), sqrt.(noise_vars)))
    return y
end

# The objective for the GP.
obj(m, x) = simulation(m, x) .- y_obs(m)

get_bounds() = ([-5., -5.], [5., 5.])


# - - - HYPERPARAMETERS - - - - -

get_kernel() = BOSS.Matern32Kernel()

const λ_MIN = 0.1
const λ_MAX = 10.
get_length_scale_priors(m) = fill(Product(fill(calc_inverse_gamma(λ_MIN, λ_MAX), 2)), y_dim(m))

function get_amplitude_priors(m)
    return fill(truncated(Normal(0., 5.); lower=0.), y_dim(m))
end

function get_noise_var_priors(m)
    return [truncated(Normal(0., (10 * ω[i])^2); lower=0.) for i in 1:y_dim(m)]
end

# get_x_prior() = Product(fill(Uniform(-5., 5.), 2))
get_x_prior() = MvNormal(zeros(2), fill(5/3, 2))


# - - - INITIALIZATION - - - - -

function get_init_data(m, count)
    X = reduce(hcat, (random_datapoint() for _ in 1:count))[:,:]
    Y = reduce(hcat, (obj(m, x) for x in eachcol(X)))[:,:]
    return BOSS.ExperimentDataPrior(X, Y)
end

bolfi_problem(init_data::Int) = bolfi_problem(get_init_data(mode, init_data))

function bolfi_problem(data::ExperimentData)
    m = mode
    return BolfiProblem(data;
        f = (x) -> obj(m, x),
        bounds = get_bounds(),
        kernel = get_kernel(),
        length_scale_priors = get_length_scale_priors(m),
        amp_priors = get_amplitude_priors(m),
        noise_var_priors = get_noise_var_priors(m),
        var_e = σe(m).^2,
        x_prior = get_x_prior(),
        y_sets = get_y_sets(m),
    )
end


# - - - UTILS - - - - -

function random_datapoint()
    x_prior = get_x_prior()
    bounds = get_bounds()

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

end
