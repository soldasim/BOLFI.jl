
module DataUtils


# --- FILES ---

using Serialization
using Distributions

const DATA_DIR = "/home/soldasim/BOLFI.jl/examples/high-dim/data/grid_B"

function load_grid_times()
    dirs = readdir(DATA_DIR)
    grid_dirs = filter(d -> startswith(d, "grid_"), dirs)
    
    dims = fname_to_dims.(grid_dirs)
    times = [deserialize(f) for f in (DATA_DIR * '/') .* grid_dirs .* ('/' * "times")]
    return dims, times
end

function fname_to_dims(fname::String)
    dims = fname[findfirst('_', fname)+1 : findfirst('-', fname)-1]
    x_dim = parse(Int, dims[findfirst('x', dims)+1 : findfirst('y', dims)-1])
    y_dim = parse(Int, dims[findfirst('y', dims)+1 : end])
    return x_dim, y_dim
end

function get_time_matrix()
    dims, times = load_grid_times()
    return get_time_matrix(dims, times)
end
function get_time_matrix(dims::AbstractVector{<:Tuple{<:Int, <:Int}}, times::AbstractVector{<:AbstractVector{<:Real}})
    xs = first.(dims)
    ys = last.(dims)

    mat = Matrix{Union{Float64, Missing}}(missing, axis_size(ys), axis_size(xs))
    for i in eachindex(dims)
        x, y = dims[i]
        ts = times[i]
        mat[y,x] = mean(ts)
    end

    return mat
end

function axis_size(indices)
    return maximum(indices) - minimum(indices) + 1
end

function get_time_ranges(dims::AbstractVector{<:Tuple{<:Int, <:Int}})
    xs = first.(dims)
    ys = last.(dims)
    return get_time_range(xs), get_time_range(ys)
end

function get_time_range(indices::AbstractVector{<:Int})
    min = minimum(indices)
    max = maximum(indices)
    
    sorted = sort(indices)
    steps = sorted[2:end] .- sorted[1:end-1]
    step = gcd(steps)
    
    return min:step:max
end


# --- PLOTS ---

using Plots

const PLOT_DIR = "/home/soldasim/BOLFI.jl/examples/high-dim/plots"

function plot_grid_times()
    dims, times = load_grid_times()
    x_range, y_range = get_time_ranges(dims)
    time_matrix = get_time_matrix(dims, times)
    
    plot_grid_times(x_range, y_range, time_matrix)
end

function plot_grid_times(x_range, y_range, time_matrix)
    p = heatmap(time_matrix ./ 60;
        x_ticks = x_range,
        y_ticks = y_range,
        xlabel = "input dim (parameters)",
        ylabel = "output dim (diagnostics)",
        colorbar_title = "time per iteration [min]",
        # colorbar_scale = :log10,
    )
    
    savefig(p, PLOT_DIR * '/' * "time_grid.png")
end

end # module DataUtils
