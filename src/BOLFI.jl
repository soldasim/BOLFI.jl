module BOLFI

export bolfi!
export estimate_parameters!, maximize_acquisition, eval_objective!
export BolfiProblem
export x_dim, y_dim

export approx_posterior, posterior_mean, posterior_variance
export approx_likelihood, likelihood_mean, likelihood_variance
export evidence
export loglike
export sample_approx_posterior, sample_posterior_mean, sample_posterior, TuringOptions
export plot_marginals_int, plot_marginals_kde, PlotSettings
export find_cutoff, approx_cutoff_area, set_iou
export get_subset

export Likelihood
export GutmannNormalLikelihood, GutmannGaussianLikelihood
export NormalLikelihood, GaussianLikelihood
export LogNormalLikelihood, LogGaussianLikelihood
export BinomialLikelihood

export BolfiAcquisition, PostVarAcq, MWMVAcq
export BolfiTermCond, AEConfidence, UBLBConfidence
export BolfiCallback, CombinedCallback
export BolfiOptions

using BOSS
using Distributions
using LinearAlgebra
using Random
using KernelFunctions
using Statistics

using StatsFuns         # used in GutmannNormalLikelihood
using SpecialFunctions  # used in utils/owent.jl: erfc

include("include.jl")

end
