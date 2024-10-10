# Version 3: GradBED inspired, but without NN

include("generalizedchisq.jl")  # TODO
using .GeneralizedChiSq

# TODO docs
struct InfoGain <: BolfiAcquisition
    samples::Int64
    Θt::Matrix{Float64}
    model_fitter::ModelFitter
end
function InfoGain(;
    samples,
    x_grid,
    model_fitter,
)
    return InfoGain(samples, x_grid, model_fitter)
end

# info gain in p(θ|D)
function (acq::InfoGain)(bolfi::BolfiProblem{Nothing}, options::BolfiOptions)
    problem = bolfi.problem
    y_dim = BOSS.y_dim(problem)

    # Sample evaluation grid deltas: Δt
    Δts, Sts = sample_Δts(problem, acq.Θt, acq.samples, bolfi.x_prior, bolfi.std_obs)
    augmented_problems = augment_problem.(Ref(problem), Ref(acq.model_fitter), Ref(acq.Θt), Δts)

    # Sample noise variables: ϵ, rand_perm
    ϵs = sample_ϵs(y_dim, acq.samples)
    rand_perm = shuffle(1:acq.samples)

    # Compute JSD
    # acq_(θ_) = jsd_function_ver1(problem, augmented_problems, rand_perm, ϵs, θ_, bolfi.std_obs, Sts)
    # acq_(θ_) = jsd_function_ver2(problem, augmented_problems, rand_perm, ϵs, θ_, bolfi.std_obs, Sts)
    acq_(θ_) = mmd_function(problem, augmented_problems, rand_perm, ϵs, θ_, bolfi.std_obs, Sts)

    return acq_
end

function calculate_St(Δt::AbstractMatrix{<:Real}, pθ::AbstractVector{<:Real}, σf::AbstractVector{<:Real})
    m = length(σf)
    Z = inv(sqrt(2π)^m * prod(σf))
    py = 1. # TODO missing

    Dt = sum(eachcol((Δt .^ 2) ./ (σf .^ 2)))
    Lt = Z .* exp.((-1/2) .* Dt)
    St = (pθ ./ py) .* Lt
    return St
end

function jsd_function_ver1(problem, augmented_problems, rand_perm, ϵs, θ_, σf, Sts)
    y_dim = BOSS.y_dim(problem)

    # prior and posteriors
    δ_prior = model_posterior(problem)
    δ_posts = model_posterior.(augmented_problems)
    μ_prior, σ_prior = δ_prior(θ_)
    μ_posts, σ_posts = [p(θ_)[1] for p in δ_posts], [p(θ_)[2] for p in δ_posts]

    # Sample δ | Δt
    δs = [μ .+ (ϵ .* σ) for (μ, σ, ϵ) in zip(μ_posts, σ_posts, ϵs)]
    
    # Calculate δ^2 | Δt samples
    δ2s = [δ .^ 2 for δ in δs]

    # Calculate d | Δt samples
    σf2 = σf .^ 2
    ds = [sum(δ2 ./ σf2) for δ2 in δ2s]
    # Calculate d
    ds_shuffled = ds[rand_perm]

    # d distributions
    ws = 1 ./ (σf .^ 2)
    ks = ones(y_dim)
    d_prior = GeneralizedChisq(ws, ks, μ_prior .^ 2, 0., 0.)
    d_posts = GeneralizedChisq.(Ref(ws), Ref(ks), [μ .^ 2 for μ in μ_posts], Ref(0.), Ref(0.))

    # Compute sample average of JSD
    vals_joint = logpdf.(d_posts, ds) .- logpdf.(Ref(d_prior), ds)
    vals_marginal = logpdf.(d_posts, ds_shuffled) .- logpdf.(Ref(d_prior), ds_shuffled)

    # val = (1/2) * ( mean((-1) .* softplus.((-1) .* vals_joint)) - mean(softplus.(vals_marginal)) )
    val = (1/2) * mean(vals_joint) - (1/2) * mean(vals_marginal)
    return val
end

function jsd_function_ver2(problem, augmented_problems, rand_perm, ϵs, θ_, σf, Sts)
    # prior and posteriors
    δ_prior = model_posterior(problem)
    δ_posts = model_posterior.(augmented_problems)
    μ_prior, σ_prior = δ_prior(θ_)
    μ_posts, σ_posts = [p(θ_)[1] for p in δ_posts], [p(θ_)[2] for p in δ_posts]

    # Sample δ | Δt
    δs = [μ .+ (ϵ .* σ) for (μ, σ, ϵ) in zip(μ_posts, σ_posts, ϵs)]
    # Sample δ
    δs_shuffled = δs[rand_perm]
    
    logpdf_(δs, μs, σs) = logpdf(MvNormal(μs, σs), δs)
    vals_joint = logpdf_.(δs, μ_posts, σ_posts) .- logpdf_.(δs, Ref(μ_prior), Ref(σ_prior))
    vals_marginal = logpdf_.(δs_shuffled, μ_posts, σ_posts) .- logpdf_.(δs_shuffled, Ref(μ_prior), Ref(σ_prior))
    
    # val = (1/2) * ( mean((-1) .* softplus.((-1) .* vals_joint)) - mean(softplus.(vals_marginal)) )
    val = (1/2) * mean(vals_joint) - (1/2) * mean(vals_marginal)
    return val
end

function mmd_function(problem, augmented_problems, rand_perm, ϵs, θ_, σf, Sts)
    # prior and posteriors
    # δ_prior = model_posterior(problem)
    δ_posts = model_posterior.(augmented_problems)
    # μ_prior, σ_prior = δ_prior(θ_)
    μ_posts, σ_posts = [p(θ_)[1] for p in δ_posts], [p(θ_)[2] for p in δ_posts]

    # Sample δ | Δt
    δs = [μ .+ (ϵ .* σ) for (μ, σ, ϵ) in zip(μ_posts, σ_posts, ϵs)]

    kernel = BOSS.GaussianKernel()
    
    joint_samples = vcat.(δs, Sts)
    marginal_samples = vcat.(δs[rand_perm], Sts)
    
    val = mmd(kernel, joint_samples, marginal_samples)
    return val
end

function sample_ϵs(y_dim::Int, samples::Int)
    dist = MvNormal(zeros(y_dim), ones(y_dim))
    ϵs = [rand(dist) for _ in 1:samples]
    return ϵs
end

function sample_Δts(problem::BossProblem, Θt::AbstractMatrix{<:Real}, samples::Int, θ_prior, σf::AbstractVector{<:Real})
    post_gps = BOSS.posterior_gp.(Ref(problem.model), Ref(problem.data), 1:BOSS.y_dim(problem))
    pred_distrs = [MvNormal(BOSS.mean_and_cov(p_gp(Θt))...) for p_gp in post_gps]

    Δt() = vcat(transpose.(rand.(pred_distrs))...)
    Δts = [Δt() for _ in 1:samples]
    # return Δts

    # TODO
    pθ = pdf.(Ref(θ_prior), eachcol(Θt))
    Sts = calculate_St.(Δts, Ref(pθ), Ref(σf))
    return Δts, Sts
end

function augment_problem(problem::BossProblem, model_fitter::ModelFitter, Θt::AbstractMatrix{<:Real}, Δt::AbstractMatrix{<:Real})
    problem_ = deepcopy(problem)
    augment_dataset!(problem_.data, Θt, Δt)
    # fit_model(problem_, model_fitter) # TODO
    return problem_
end

function fit_model(problem::BossProblem, model_fitter::ModelFitter)
    options = BossOptions(;
        info = false,
    )
    BOSS.estimate_parameters!(problem, model_fitter; options)
end

softplus(x) = log(one(x) + exp(x))

function mmd(kernel, X::AbstractVector, Y::AbstractVector)
    val_X = mean(BOSS.kernelmatrix(kernel, X))
    val_Y = mean(BOSS.kernelmatrix(kernel, Y))
    val_XY = mean(BOSS.kernelmatrix(kernel, X, Y))
    return val_X + val_Y - 2*val_XY
end
