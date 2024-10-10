module BOLFI

export bolfi!
export BolfiProblem

export approx_posterior, posterior_mean, posterior_variance, evidence
export gp_mean, gp_bound, gp_quantile
export find_cutoff, approx_cutoff_area
export get_subset
export set_iou

export BolfiAcquisition, PostVarAcq, MWMVAcq
export BolfiTermCond, AEConfidence, UBLBConfidence
export JSDivergence, MaximumMeanDiscrepancy, OptTransport, LikelihoodStd # TODO
export BolfiCallback
export BolfiOptions

export GaussMixOptions

using BOSS
using Distributions
using Distances
using OptimalTransport

include("include.jl")

end
