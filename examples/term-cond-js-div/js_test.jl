include("main.jl")

p1 = MvNormal([-1.], [5.])
p2 = MvNormal([1.], [5.])

post_1(x) = pdf(p1, x)
post_2(x) = pdf(p2, x)

# q1 = p1
# q2 = p2
q1 = q2 = MvNormal([0.], [10.])

samples = 1
BOLFI.jensen_shannon_divergence(post_1, post_2, q1, q2; samples)
