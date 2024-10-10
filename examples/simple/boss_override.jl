
@warn "Overriding some methods in BOSS!"

using LinearAlgebra

function BOSS.estimate_parameters(opt::SamplingMAP, problem::BossProblem, options::BossOptions; return_all::Bool=false)
    sample_() = BOSS.sample_params(problem.model)
    fitness = BOSS.model_loglike(problem.model, problem.data)

    # - - - - - - - -
    function fitness_(params...)
        val = -Inf
        try
            val = fitness(params...)
        catch e
            if e isa PosDefException
                @warn `PosDefException`
            else
                throw(e)
            end
        end
        return val
    end
    # - - - - - - - -

    params, fit = BOSS.sample(Val(return_all), Val(opt.parallel), opt, sample_, fitness_)
    return params, fit
end
