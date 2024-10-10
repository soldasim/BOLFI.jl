
# - - - INFO GAIN IN θ (not what we want) - - - - -

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
    
    # GP posterior given current data
    gp_old = BOSS.model_posterior(boss.model, boss.data)

    # old neg. entropy (Does not depend on `x`.)
    info_old = neg_entropy_(gp_old, acq.x_grid, bolfi.x_prior)

    function mi(x)
        μ_y, σ_y = gp_old(x)
        y_samples = acq.ϵ_samples .* σ_y .+ μ_y
        
        # GP posteriors given augmented data
        gps = BOSS.model_posterior.(
            Ref(boss.model),
            BOSS.augment_dataset!.(
                [deepcopy(boss.data) for _ in 1:acq.samples],
                Ref(x),
                eachcol(y_samples),
            ),
        )
        
        # new neg. entropy
        info = neg_entropy_.(gps, Ref(acq.x_grid), Ref(bolfi.x_prior)) |> mean

        # info. gain
        ig = info - info_old
        return ig
    end
end

function neg_entropy_(gp, x_grid, x_prior)
    function loglike(θ)
        μ, σ = gp(θ)
        return logpdf(MvNormal(μ, σ), zeros(length(μ)))
    end

    log_lls = loglike.(eachcol(x_grid))              # log l_j
    log_pxs = logpdf.(Ref(x_prior), eachcol(x_grid)) # log p_j

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
