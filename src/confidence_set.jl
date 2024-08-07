
"""
Return predictive function `(x) -> (m, 0.)`, where `m` is the mean
of the GP posterior.
"""
function gp_mean(gp_post)
    return function post(x)
        m, std = gp_post(x)
        std_ = fill(0., length(m))
        return m, std_
    end
end

"""
Return predictive function `(x) -> (m, 0.)`, where `m = μ + n * std`,
where `μ` and `std` are the mean and the standard deviation of the GP posterior.
"""
function gp_bound(gp_post, n)
    return function post(x)
        m, std = gp_post(x)
        m_ = m .+ (n * std)
        std_ = fill(0., length(m))
        return m_, std_
    end
end

"""
Return predictive function `(x) -> (m, 0.)`, where `m` is the `q`th quantile
of the posterior predictive distribution of the GP posterior.
"""
function gp_quantile(gp_post, q)
    return function post(x)
        m, std = gp_post(x)
        d = Normal.(m, std)
        m_ = quantile.(d, Ref(q))
        std_ = fill(0., length(m))
        return m_, std_
    end
end

"""
Returns the provided density function together with cutoff `c`
s.t. the ratio `q` of probability mass
lies within the area given by `{x | post(x) > c}`.
"""
function find_cutoff(post, x_prior, q; xs=nothing, samples=10_000)
    isnothing(xs) && (xs = rand(x_prior, samples))
    ws = post.(eachcol(xs)) ./ pdf.(Ref(x_prior), eachcol(xs))
    vals = post.(eachcol(xs))
    c = quantile(vals, Distributions.weights(ws), 1. - q)
    return post, c
end

"""
Approximate the ratio of the area where `post(x) > c` relative to the whole support of `post(x)`.
(The prior `x_prior` must support the whole support of `post(x)`.)
"""
function approx_cutoff_area(post, x_prior, c; xs=nothing, samples=10_000)
    if isnothing(xs)
        xs = rand(x_prior, samples)
    end
    ws = 1 ./ pdf.(Ref(x_prior), eachcol(xs))
    ws ./= sum(ws)
    V = sum(ws[post.(eachcol(xs)) .> c])
    return V
end

"""
Approximate the intersection-over-union of two sets A and B.

The parameters `in_A`, `in_B` are binary arrays declaring which samples
from `xs` fall into the sets A and B. The matrix `xs` contains the samples
in its columns. The samples have to be drawn from the common prior `x_prior`.
"""
function set_iou(in_A, in_B, x_prior, xs)
    isnothing(xs) && (xs = rand(x_prior, samples))

    ws = 1 ./ pdf.(Ref(x_prior), eachcol(xs))
    ws ./= sum(ws)

    V_intersect = sum(ws[in_A .&& in_B])
    V_union = sum(ws[in_A .|| in_B])
    return V_intersect / V_union
end
