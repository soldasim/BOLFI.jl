
# - - - INFO GAIN via Laplace approx. of δ^2 distribution - - - - -

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

function (acq::InfoGain)(bolfi::BolfiProblem{Nothing}, options::BolfiOptions)
    boss = bolfi.problem

    gp_posts(model, data) = BOSS.posterior_gp.(Ref(model), Ref(data), 1:BOSS.y_dim(data))

    gp_old = BOSS.model_posterior(boss.model, boss.data)
    gp_posts_old = gp_posts(boss.model, boss.data)

    info_old = neg_entropy_(gp_posts_old, acq.x_grid, acq.y_samples[1], bolfi.std_obs)

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
        info = neg_entropy_.(gp_posts_new, Ref(acq.x_grid), Ref(acq.y_samples[1]), Ref(bolfi.std_obs)) |> mean

        # info. gain
        ig = info - info_old
        return ig
    end
end

function neg_entropy_(
    gp_posts,
    X_grid::AbstractMatrix{<:Real},
    Y_samples::AbstractMatrix{<:Real},  #unused here
    std_obs::AbstractVector{<:Real},
)
    # 1) Get dist. of δ from GP.
    # 2) Estimate true dist. of δ^2 (non-centered chi-squared) with approximate normal.
    #       μ_ ≈ 1 + μ^2, σ_ ≈ sqrt( 2(1 + 2μ^2) )
    # 3) Analytically derive log-normal dist of likelihood value
    #    and analytically calculate its entropy.

    δ_dists = [MvNormal(BOSS.mean_and_cov(post(X_grid))...) for post in gp_posts]

    # TODO treating different grid points as independent for now
    μs = mapreduce(d -> (d.μ)', vcat, δ_dists)
    # σs = mapreduce(d -> diag(d.Σ)', vcat, δ_dists) # we lost dependence on δ variance in the approximation

    As = (-1) * ((1 .+ μs .^ 2) ./ (2 * std_obs .^ 2) .+ log.(sqrt(2π) * std_obs))
    Bs = (sqrt.(2 * (1 .+ 2 * μs .^ 2))) ./ (2 * std_obs .^ 2)

    return (-1) * mapreduce(t -> entropy(LogNormal(t...)), +, zip(As, Bs))
end
