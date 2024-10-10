
# - - - INFOGAIN BY SAMPLING & VARIABLE TRANSFORMATION (jacobian etc) - - - - -
# Version 2: Chi squared stuff ...

include("generalizedchisq.jl")  # TODO
using .GeneralizedChiSq

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
    x_grid::Matrix{Float64},
    y_dim::Int,
    samples::Int,
)
    grid_size = size(x_grid)[2]
    y_samples = [rand(MvNormal(zeros(y_dim), ones(y_dim)), grid_size) for _ in 1:samples]
    ϵ_samples = rand(MvNormal(zeros(y_dim), ones(y_dim)), samples)
    return InfoGain(samples, x_grid, y_samples, ϵ_samples)
end

# info gain in p(θ|D)
function (acq::InfoGain)(bolfi::BolfiProblem{Nothing}, options::BolfiOptions)
    boss = bolfi.problem
    # TODO
    # neg_entropy_ = neg_entropy_nonlog_
    neg_entropy_ = neg_entropy_log_

    gp_posts(model, data) = BOSS.posterior_gp.(Ref(model), Ref(data), 1:BOSS.y_dim(data))

    gp_old = BOSS.model_posterior(boss.model, boss.data)
    gp_posts_old = gp_posts(boss.model, boss.data)
    info_old = neg_entropy_.(Ref(gp_posts_old), Ref(acq.x_grid), acq.y_samples, Ref(bolfi.std_obs), Ref(bolfi.x_prior)) |> mean

    function mi(x)
        μ_y, σ_y = gp_old(x)
        ys = acq.ϵ_samples .* σ_y .+ μ_y
        
        # GP posteriors given augmented data
        # (Sample different `y_` for the `x_` being evaluated.)
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

# LOG VERSION
"""
Calculate single neg. entropy sample.
"""
function neg_entropy_log_(
    gp_posts,
    X_grid::AbstractMatrix{<:Real},
    Y_samples::AbstractMatrix{<:Real},
    std_obs::AbstractVector{<:Real},
    x_prior::Distribution,
)
    y_dim = length(std_obs)
    grid_size = size(X_grid)[2]


    # - - - SAMPLING - - - - -

    update_gp_post(gp_post, x_, δ_) = BOSS.posterior(gp_post(x_), [δ_])

    # Compute δs using the chain rule.
    δ = Matrix{Float64}(undef, y_dim, grid_size)
    λ = Matrix{Float64}(undef, y_dim, grid_size)  # for prob. calc. later

    for i in 1:grid_size
        # sample δ_i
        for slice in 1:y_dim
            μ, σ2 = first.(BOSS.mean_and_var(gp_posts[slice](X_grid[:,i])))
            δ[slice,i] = μ + sqrt(σ2) * Y_samples[slice,i]
            λ[slice,i] = μ^2
        end
        # update GP posteriors
        gp_posts = update_gp_post.(gp_posts, Ref(X_grid[:,i]), δ[:,i])
    end

    # Compute s.
    s(i) = sum((δ[:,i] ./ std_obs) .^ 2)

    log_Z = (-1) * ((y_dim/2)*log(2π) + sum(log.(std_obs)))
    log_l(i) = log_Z - (s(i) / 2)

    log_pθ = logpdf.(Ref(x_prior), eachcol(X_grid))
    pθ_sum = sum(exp.(log_pθ))
    evidence = sum(exp.(log_l.(1:grid_size) .+ log_pθ .- log(pθ_sum)))

    # p(i) = (pdf(x_prior, X_grid[:,i]) / evidence) * l(i)
    

    # - - - PROB. COMPUATATION - - - - -
    
    w = 1 ./ (std_obs .^ 2)
    k = fill(1, y_dim)
    # λ  # from before
    gchisq(i) = GeneralizedChisq(w, k, λ[:,i], 0., 0.)

    log_s_wrt_l(i) = log(2) - log_l(i)
    log_l_wrt_p(i) = log(evidence) - logpdf(x_prior, X_grid[:,i])

    log_p_p(i) = logpdf(gchisq(i), s(i)) + log_s_wrt_l(i) + log_l_wrt_p(i)

    return sum(log_p_p.(1:grid_size))
end

# NON-LOG VERSION
"""
Calculate single neg. entropy sample.
"""
function neg_entropy_nonlog_(
    gp_posts,
    X_grid::AbstractMatrix{<:Real},
    Y_samples::AbstractMatrix{<:Real},
    std_obs::AbstractVector{<:Real},
    x_prior::Distribution,
)
    y_dim = length(std_obs)
    grid_size = size(X_grid)[2]


    # - - - SAMPLING - - - - -

    update_gp_post(gp_post, x_, δ_) = BOSS.posterior(gp_post(x_), [δ_])

    # compute δs using the chain rule
    δ = Matrix{Float64}(undef, y_dim, grid_size)
    λ = Matrix{Float64}(undef, y_dim, grid_size)  # for prob. calc. later

    for i in 1:grid_size
        # sample δ_i
        for slice in 1:y_dim
            μ, σ2 = first.(BOSS.mean_and_var(gp_posts[slice](X_grid[:,i])))
            δ[slice,i] = μ + sqrt(σ2) * Y_samples[slice,i]
            λ[slice,i] = μ^2
        end
        # update GP posteriors
        gp_posts = update_gp_post.(gp_posts, Ref(X_grid[:,i]), δ[:,i])
    end

    s(i) = sum((δ[:,i] ./ std_obs) .^ 2)

    Z = 1 / ((sqrt(2π) ^ y_dim) * prod(std_obs))
    l(i) = Z * exp(- (1/2) * s(i))

    pθ = pdf.(Ref(x_prior), eachcol(X_grid))
    pθ_sum = sum(pθ)
    evidence = sum(l.(1:grid_size) .* (pθ / pθ_sum))

    # p(i) = (pdf(x_prior, X_grid[:,i]) / evidence) * l(i)
    

    # - - - PROB. COMPUATATION - - - - -
    
    w = 1 ./ (std_obs .^ 2)
    k = fill(1, y_dim)
    # λ  # from before
    gchisq(i) = GeneralizedChisq(w, k, λ[:,i], 0., 0.)

    s_wrt_l(i) = abs((-2) / l(i))
    l_wrt_p(i) = abs(evidence / pdf(x_prior, X_grid[:,i]))

    p_p(i) = pdf(gchisq(i), s(i)) * s_wrt_l(i) * l_wrt_p(i)

    return sum(log.(p_p.(1:grid_size)))
end
