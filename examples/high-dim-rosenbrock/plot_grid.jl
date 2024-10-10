module PlotGrid

using Serialization
using Plots
using Distributions

const DIR = "./examples/high-dim-rosenbrock/data/grid-cost"
const SAVE_DIR = "./examples/high-dim-rosenbrock/plots/cost-grid"

# - - - MAIN SCRIPTS - - -

function plot_success(; save=true)
    success(idx, times, vals) = sum((!isnothing).(vals))
    
    G = load_grid(success)
    G[isnothing.(G)] .= -5.

    p = heatmap(G; title="successes out of 20", xlabel="input dim (parameters)", ylabel="output dim (diagnostics)")
    display(p)
    save && savefig(p, SAVE_DIR * '/' * "success.png")
end

function plot_costs(; save=true)
    function cost(idx, times, vals)
        vals = filter(!isnothing, vals)
        isempty(vals) && return nothing
        return mean(vals)
    end
    
    G = load_grid(cost)
    G[isnothing.(G)] .= -1.

    p = heatmap(G; title="mean cost", xlabel="input dim (parameters)", ylabel="output dim (diagnostics)")
    display(p)
    save && savefig(p, SAVE_DIR * '/' * "costs.png")
end

function plot_cost_std(; save=true)
    function cost_std(idx, times, vals)
        vals = filter(!isnothing, vals)
        isempty(vals) && return nothing
        return std(vals)
    end
    
    G = load_grid(cost_std)
    G[isnothing.(G)] .= -0.1

    p = heatmap(G; title="mean time", xlabel="input dim (parameters)", ylabel="output dim (diagnostics)")
    display(p)
    save && savefig(p, SAVE_DIR * '/' * "cost_std.png")
end

function plot_times(; save=true)
    function time(idx, times, vals)
        return mean(times)
    end
    
    G = load_grid(time)
    G[isnothing.(G)] .= -1.

    p = heatmap(G; title="mean time", xlabel="input dim (parameters)", ylabel="output dim (diagnostics)")
    display(p)
    save && savefig(p, SAVE_DIR * '/' * "times.png")
end

# - - - UTILS - - -

function load_grid(value_getter)
    fs = DIR .* '/' .* readdir(DIR)
    points = load_gridpoint.(fs)
    x_lims, y_lims = grid_size(points)
    
    G = Matrix{Union{Nothing, Float64}}(nothing, dim_size(x_lims), dim_size(y_lims))
    for p in points
        idx = p[1]
        G[idx[2], idx[1]] = value_getter(p...)
    end
    return G
end

dim_size(lims) = lims[2] - lims[1] + 1

function grid_size(points)
    idxs = getindex.(points, Ref(1))
    x_dims = getindex.(idxs, Ref(1))
    y_dims = getindex.(idxs, Ref(2))
    x_min, x_max = minimum(x_dims), maximum(x_dims)
    y_min, y_max = minimum(y_dims), maximum(y_dims)
    return (x_min, x_max), (y_min, y_max)
end

function load_gridpoint(subdir)
    idx = get_idx(subdir)
    vals = get_vals(subdir)
    times = get_times(subdir)
    return idx, times, vals
end

function get_idx(subdir)
    dirname = subdir[findlast('/', subdir)+1:end]
    cols = findall('-', dirname)
    @assert length(cols) == 2
    x_dim = parse(Int, dirname[cols[1]+1:cols[2]-1])
    y_dim = parse(Int, dirname[cols[2]+1:end])
    return x_dim, y_dim
end

get_vals(subdir) = deserialize(subdir * '/' * "vals")
get_times(subdir) = deserialize(subdir * '/' * "times")

end # module PlotGrid
