
"""
    BolfiOptions(; kwargs...)

Stores miscellaneous settings.

# Keywords
- `info::Bool`: Setting `info=false` silences the algorithm.
- `debug::Bool`: Set `debug=true` to print stactraces of caught optimization errors.
- `parallel_evals::Symbol`: Possible values: `:serial`, `:parallel`, `:distributed`. Defaults to `:parallel`.
        Determines whether to run multiple objective function evaluations
        within one batch in serial, parallel, or distributed fashion.
        (Only has an effect if batching AM is used.)
- `callback::Union{<:BossCallback, <:BolfiCallback}`: If provided,
        the callback will be called before the BO procedure starts and after every iteration.
"""
@kwdef struct BolfiOptions{
    CB<:Union{<:BossCallback, <:BolfiCallback},
}
    info::Bool = true
    debug::Bool = false
    parallel_evals::Symbol = :parallel
    callback::CB = NoCallback()
end

create_boss_options(opt::BolfiOptions, bolfi::BolfiProblem) = BOSS.BossOptions(
    opt.info,
    opt.debug,
    opt.parallel_evals,
    CallbackWrapper(opt.callback, bolfi),
)
