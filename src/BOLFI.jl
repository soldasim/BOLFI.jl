module BOLFI

export bolfi!
export BolfiProblem

export get_subset
export posterior_mean, posterior_variance, evidence
export gp_mean, gp_bound, gp_quantile
export find_cutoff, find_cutoff_max, set_overlap

export BolfiAcquisition, PostVariance, SetsPostVariance
export BolfiTermCond, ConfidenceTermCond, UBLBConfidence
export BolfiCallback
export BolfiOptions

using BOSS
using Distributions

include("problem.jl")
include("subset_problem.jl")
include("options.jl")
include("posterior.jl")
include("confidence_set.jl")
include("acquisition.jl")
include("term_cond.jl")
include("main.jl")

end
