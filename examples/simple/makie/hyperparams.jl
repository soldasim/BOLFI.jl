
function plot_lengthscales(bolfi, acquisition, grid)
    @info "Plotting ..."

    powers = collect(-4:2:4)
    lengthscales = 10. .^ powers
    
    acq_samples = 4
    acqs = [acquisition(bolfi, BolfiOptions()) for _ in 1:acq_samples]

    # --- PLOT ---
    f = Figure(;
        size = (1600, 1600),
    )

    for i in eachindex(lengthscales), j in eachindex(lengthscales)
        δ_scale = lengthscales[i]
        S_scale = lengthscales[j]
        
        ax = Axis(f[i,j]; title="δ_scale = 1e$(powers[i]), S_scale = 1e$(powers[j])")
        for a in acqs
            vals = calculate_values(x -> a([x]; δ_scale, S_scale), grid)
            lines!(ax, grid, vals)
        end
        scatter!(ax, vec(bolfi.problem.data.X), zeros(length(bolfi.problem.data.X)); label = "data")
        vlines!(ax, [ToyProblem.y_obs[1]]; linestyle=:dash)
    end

    return f
end
