
"""
Selects the next evaluation point by maximizing the mutual information
of the parameters and the data.
"""
struct InfoGain <: BolfiAcquisition
    x_samples::Matrix{Float64}
    ϵ_samples::Matrix{Float64}
    parallel::Bool
end
function InfoGain(;
    x_prior::MultivariateDistribution,
    y_dim::Int,
    samples::Int,
    parallel::Bool = true,
)
    x_samples = rand(x_prior, samples)
    ϵ_samples = rand(MvNormal(zeros(y_dim), ones(y_dim)), samples)
    return InfoGain(x_samples, ϵ_samples, parallel)
end

function (acq::InfoGain)(bolfi::BolfiProblem{Nothing}, options::BolfiOptions)
    @assert size(acq.x_samples)[2] == size(acq.ϵ_samples)[2]
    samples = size(acq.x_samples)[2]
    boss = bolfi.problem
    
    # GP posterior given current data
    gp_old = BOSS.model_posterior(boss.model, boss.data)

    # old neg. entropy (Does not depend on `x`.)
    info_old = neg_entropy_(gp_old, acq.x_samples, bolfi.x_prior)

    function mi(x)
        μ_y, σ_y = gp_old(x)
        y_samples = acq.ϵ_samples .* σ_y .+ μ_y
        
        # GP posteriors given augmented data
        gps = BOSS.model_posterior.(
            Ref(boss.model),
            BOSS.augment_dataset!.(
                [deepcopy(boss.data) for _ in 1:samples],
                Ref(x),
                eachcol(y_samples),
            ),
        )
        
        # new neg. entropy
        if acq.parallel
            vals = Vector{Float64}(undef, length(gps))
            Threads.@threads :static for i in eachindex(gps)
                vals[i] = neg_entropy_(gps[i], acq.x_samples, bolfi.x_prior)
            end
            info = mean(vals)
        else
            info = neg_entropy_.(gps, Ref(acq.x_samples), Ref(bolfi.x_prior)) |> mean
        end

        # info. gain
        ig = info - info_old
        return ig
    end
end

function neg_entropy_(gp, x_samples, x_prior)
    function loglike(θ)
        μ, σ = gp(θ)
        return logpdf(MvNormal(μ, σ), zeros(length(μ)))
    end

    log_lls = loglike.(eachcol(x_samples))              # log l_j
    log_pxs = logpdf.(Ref(x_prior), eachcol(x_samples)) # log p_j

    M = maximum(log_lls)
    w_vals = exp.(log_lls .- M)
    py = mean(w_vals)   # e
    ws = w_vals ./ py   # w_j

    A = ws .* log.(ws)
    A[ws .== 0.] .= 0.  # (w = 0) => (w log w = 0)
    B = ws .* log_pxs
    vals = A .+ B

    return mean(vals)
end
