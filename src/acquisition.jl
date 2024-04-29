
"""
An abstract type for BOLFI acquisition functions.

# Creating custom acquisition function for BOLFI:
- Create type `CustomAcq <: BolfiAcquisition`
- Implement method `(::CustomAcq)(::BolfiProblem, ::BolfiOptions) -> (x -> ::Real)`
"""
abstract type BolfiAcquisition end

struct AcqWrapper{
    A<:BolfiAcquisition
} <: AcquisitionFunction
    acq::A
    bolfi::BolfiProblem
    options::BolfiOptions
end

(wrap::AcqWrapper)(::BossProblem, ::BossOptions) = wrap.acq(wrap.bolfi, wrap.options)


# - - - Posterior Variance - - - - -

struct PostVariance <: BolfiAcquisition end

function (acq::PostVariance)(bolfi::BolfiProblem{Nothing}, options::BolfiOptions)
    problem = bolfi.problem
    @assert problem.data isa BOSS.ExperimentDataMLE
    gp_post = BOSS.model_posterior(problem.model, problem.data)
    return posterior_variance(bolfi.x_prior, gp_post, bolfi.var_e; normalize=false)
end


# - - - Posterior Variance for Observation Sets - - - - -

struct SetsPostVariance <: BolfiAcquisition
    samples::Int
end
SetsPostVariance(;
    samples = 10_000,
) = SetsPostVariance(samples)

function (acq::SetsPostVariance)(bolfi::BolfiProblem{Matrix{Bool}}, options::BolfiOptions)
    problem = bolfi.problem
    @assert problem.data isa BOSS.ExperimentDataMLE

    gp_posts = BOSS.model_posterior(problem.model, problem.data; split=true)
    xs = rand(bolfi.x_prior, acq.samples)  # shared samples
    
    set_vars = Vector{Function}(undef, size(bolfi.y_sets)[2])
    for i in eachindex(set_vars)
        set = bolfi.y_sets[:,i]
        post = combine_gp_posts(gp_posts[set])
        var_e = bolfi.var_e[set]
        set_vars[i] = posterior_variance(bolfi.x_prior, post, var_e; normalize=true, xs)
    end
    return (x) -> mean((v(x) for v in set_vars))
end

function combine_gp_posts(gp_posts)
    function post(x)
        preds = [post(x) for post in gp_posts]
        first.(preds), last.(preds)
    end
end
