
# - - - INFOGAIN BY SAMPLING & VARIABLE TRANSFORMATION (jacobian etc) - - - - -
# TODO: Works only for y_dim == 1.

"""
Selects the next evaluation point by maximizing the mutual information
of the parameters and the data.
"""
struct InfoGain <: BolfiAcquisition
    samples::Int
    x_grid::Matrix{Float64}             # grid which the GP is discretized to
    y_samples::Vector{Matrix{Float64}}  # ϵ samples for the grid for each y_new sample
    ϵ_samples::Matrix{Float64}          # ϵ samples for the y_new
end
function InfoGain(;
    x_prior::MultivariateDistribution,
    y_dim::Int,
    samples::Int,
)
    x_grid = rand(x_prior, samples)
    y_samples = [rand(MvNormal(zeros(y_dim), ones(y_dim)), samples) for _ in 1:samples]
    ϵ_samples = rand(MvNormal(zeros(y_dim), ones(y_dim)), samples)
    return InfoGain(samples, x_grid, y_samples, ϵ_samples)
end

# info gain in p(θ|D)
function (acq::InfoGain)(bolfi::BolfiProblem{Nothing}, options::BolfiOptions)
    boss = bolfi.problem

    gp_posts(model, data) = BOSS.posterior_gp.(Ref(model), Ref(data), 1:BOSS.y_dim(data))

    gp_old = BOSS.model_posterior(boss.model, boss.data)
    gp_posts_old = gp_posts(boss.model, boss.data)
    info_old = neg_entropy_.(Ref(gp_posts_old), Ref(acq.x_grid), acq.y_samples, Ref(bolfi.std_obs), Ref(bolfi.x_prior)) |> mean

    function mi(x)
        μ_y, σ_y = gp_old(x)
        ys = acq.ϵ_samples .* σ_y .+ μ_y
        
        # GP posteriors given augmented data
        gp_posts_new = gp_posts.(
            Ref(boss.model),
            BOSS.augment_dataset!.(
                [deepcopy(boss.data) for _ in 1:acq.samples],
                Ref(x),
                eachcol(ys),
            ),
        )
        
        # new neg. entropy
        # single Y samples for each new gp post
        info = neg_entropy_.(gp_posts_new, Ref(acq.x_grid), acq.y_samples, Ref(bolfi.std_obs), Ref(bolfi.x_prior)) |> mean

        # info. gain
        ig = info - info_old
        return ig
    end
end

function neg_entropy_(
    gp_posts,
    X_grid::AbstractMatrix{<:Real},
    Y_samples::AbstractMatrix{<:Real},
    std_obs::AbstractVector{<:Real},
    x_prior::Distribution,
)
    # TODO the maths works for y_dim = 1 only
    @assert (length(gp_posts) == 1) && (size(Y_samples)[1] == 1) && (length(std_obs) == 1)
    gp_post, y_samples, std_obs = gp_posts[1], Y_samples[1,:], std_obs[1]

    post = MvNormal(BOSS.mean_and_cov(gp_post(X_grid))...)
    μ = post.μ
    L = factorize(post.Σ).L
    y_grid = μ + L * y_samples

    # l = pdf.(Ref(Normal(0., std_obs)), y_grid)  # likelihood
    # pθ = pdf.(Ref(x_prior), eachcol(X_grid))    # prior
    # py = mean(l)                                # evidence (assumes X_grid ~ x_prior)
    # # p = pθ .* (l ./ py)                       # posterior

    # y_wrt_l(y, l) = (std_obs^2) / (y * l) |> abs
    # l_wrt_p(pθ) = py / pθ |> abs

    # return pdf(post, y_grid) * prod(y_wrt_l.(y_grid, l) .* l_wrt_p.(pθ)) |> log

    log_l = logpdf.(Ref(Normal(0., std_obs)), y_grid)  # likelihood
    log_pθ = logpdf.(Ref(x_prior), eachcol(X_grid))    # prior
    log_py = log(mean(exp.(log_l)))                    # evidence (assumes X_grid ~ x_prior)
    # log_p = log_pθ .+ log_l .- log_py                # posterior

    log_y_wrt_l(y, log_l) = 2*log(std_obs) - (log(abs(y)) + log_l)
    log_l_wrt_p(log_pθ) = log_py - log_pθ

    return logpdf(post, y_grid) + sum(log_y_wrt_l.(y_grid, log_l) .+ log_l_wrt_p.(log_pθ))
end
