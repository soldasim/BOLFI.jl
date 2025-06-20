
function plot_state_2d(bolfi, new_datum; plt, sample_posterior=true, kwargs...)
    boss = bolfi.problem
    acquisition = unwrap(bolfi.problem.acquisition)

    approx_post = approx_posterior(bolfi)
    post_mean = posterior_mean(bolfi)
    post_var = posterior_variance(bolfi)
    gp_post = BOSS.model_posterior(boss)

    acq_samples = 4
    acqs = [acquisition(bolfi, BolfiOptions()) for _ in 1:acq_samples]
    
    # --- PLOTS ---
    f = Figure(;
        # size = (1440, 810),
        size = (1440, 1620),
    )

    ax = Axis(f[1,1];
        title = "true posterior",
    )
    plot_func_2d!(ax, (x,y) -> true_post([x,y]), bolfi, new_datum; plt)
    plot_data_2d!(ax, bolfi, new_datum; plt)
    # axislegend(ax; position = :rt)

    ax = Axis(f[1,2];
        title = "approx. posterior",
    )
    plot_func_2d!(ax, (x,y) -> approx_post([x,y]), bolfi, new_datum; plt)
    sample_posterior && plot_posterior_samples!(ax, approx_post, bolfi)
    plot_data_2d!(ax, bolfi, new_datum; plt)
    # axislegend(ax; position = :rt)

    # ax = Axis(f[2,1];
    #     title = "abs. val. of GP mean",
    # )
    # plot_func_2d!(ax, (x,y) -> abs(gp_post([x,y])[1][1]), bolfi, new_datum; plt)
    # plot_data_2d!(ax, bolfi, new_datum; plt)
    # # axislegend(ax; position = :rt)

    ax = Axis(f[2,1];
        title = "posterior mean",
    )
    plot_func_2d!(ax, (x,y) -> post_mean([x,y]), bolfi, new_datum; plt)
    plot_data_2d!(ax, bolfi, new_datum; plt)
    # axislegend(ax; position = :rt)

    ax = Axis(f[2,2];
        title = "posterior variance",
    )
    plot_func_2d!(ax, (x,y) -> post_var([x,y]), bolfi, new_datum; plt, label="posterior variance", grid=plt.acq_grid)
    plot_data_2d!(ax, bolfi, new_datum; plt)
    # axislegend(ax; position = :rt)

    return f
end

function plot_func_2d!(ax, func, bolfi, new_datum; plt, label=nothing, grid=nothing, normalize=false, kwargs...)
    xs, ys = isnothing(grid) ? get_xs_2d(bolfi; plt) : (grid, grid)
    
    if plt.parallel
        vals = calculate_values(t -> func(t...), Iterators.product(xs, ys))
    else
        vals = map(t -> func(t...), Iterators.product(xs, ys))
    end
    
    normalize && normalize_values!(vals)
    contourf!(ax, xs, ys, vals;
        label,
        kwargs...
    )
end

function plot_data_2d!(ax, bolfi, new_datum; plt)
    boss = bolfi.problem

    scatter!(ax, boss.data.X[1,:], boss.data.X[2,:];
        label = "data",
        color = :cyan,
    )
    isnothing(new_datum) || scatter!(ax, [new_datum[1]], [new_datum[2]];
        label = "new datum",
        color = :orange,
    )
end

function plot_posterior_samples!(ax, post, bolfi)
    bounds = bolfi.problem.domain.bounds
    xs = BOLFI.sample_posterior(x -> log(post(x)), bounds)

    # fix limits so that samples out of bounds are not plotted
    xlims = bounds[1][1], bounds[2][1]
    ylims = bounds[1][2], bounds[2][2]
    xlims!(ax, xlims)
    ylims!(ax, ylims)

    scatter!(ax, xs;
        color = :green,
        marker = :xcross,
        markersize = 6,
    )
end

function get_xs_2d(bolfi; plt)
    lb1, ub1 = first.(bolfi.problem.domain.bounds)
    lb2, ub2 = getindex.(bolfi.problem.domain.bounds, Ref(2))
    return lb1:plt.step:ub1, lb2:plt.step:ub2
end
