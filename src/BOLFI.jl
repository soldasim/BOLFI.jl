module BOLFI

export bolfi!
export estimate_parameters!, maximize_acquisition, eval_objective!
export BolfiProblem

export sample_posterior, TuringOptions
export plot_marginals_int, plot_marginals_kde, PlotSettings
export approx_posterior, posterior_mean, posterior_variance
export approx_likelihood, likelihood_mean, likelihood_variance
export evidence
export find_cutoff, approx_cutoff_area, set_iou
export get_subset

export BolfiAcquisition, PostVarAcq, MWMVAcq, InfoGain
export BolfiTermCond, AEConfidence, UBLBConfidence
export BolfiCallback, CombinedCallback
export BolfiOptions

using BOSS
using Distributions
using LinearAlgebra
using Random
using KernelFunctions
using Statistics

include("include.jl")

end
