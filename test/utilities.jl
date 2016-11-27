"Random simplex of `n` elements, generated by normalizing draws from
`distribution`."
rand_simplex(n, distribution) = normalize(abs(rand(distribution, n)), 1)

"Random sparse stochastic matrix with size (nrow,ncol).

Matrix has `m` nonzerelements in each row, which are generated by
normalizing the absolute value of draws from `distribution`."
function random_sparse_stochastic_matrix(nrow::Int, m::Int;
                                         distribution = Normal(),
                                         ncol = nrow)
    I = repeat(1:nrow, inner=m)
    J = vcat([sample(1:ncol, m, replace=false) for i in 1:nrow]...)
    V = vcat([rand_simplex(m, distribution) for i in 1:nrow]...)
    sparse(I, J, V, nrow, ncol)
end

function random_stochastic_matrix(nrow; distribution = Normal(), ncol = nrow)
    P = Array(Float64, nrow, ncol)
    for i in 1:nrow
        P[i, :] = rand_simplex(ncol, distribution)
    end
    P
end

"""
Simulate a path of length `T` from HMC (P,Q), with initial distribution `p`.
"""
function simulate_path(p::AbstractVector, P::AbstractMatrix,
                           Q::AbstractMatrix, T::Int)
    _rand(v) = rand(Categorical(v))
    _rand(M, i) = _rand(collect(M[i, :]))
    i = _rand(p)
    path = Array(Int64, T)
    for t in 1:T
        path[t] = _rand(Q, i)
        if t != T
            i = _rand(P, i)
        end
    end
    path
end

"""
Simulated probabilities for paths, using `n` draws. See
`simulate_path` for the other parameters.
"""
function simulate_path_probabilities(p::AbstractVector, P::AbstractMatrix,
                                     Q::AbstractMatrix, T::Int, n::Int)
    counts = Dict{Array{Int64,1},Int64}()
    for i in 1:n
        path = simulate_path(p, P, Q, T)
        counts[path] = get(counts, path, 0) + 1
    end
    Dict(path => count/n for (path, count) in counts)
    end

"""
Average and largest absolute difference in probabilities. Iterates
over paths in `pp`.
"""
function path_probabilities_difference(pp, pp_sim)
    sumdiff = 0.0
    maxdiff = 0.0
    for (path, prob) in pp
        diff = abs(pp_sim[path] - prob)
        sumdiff += diff
        maxdiff = max(maxdiff, diff)
    end
    (sumdiff/length(pp), maxdiff)
end

"""
Paths in simulation which are not covered by exact calculations. A
nonempty set may indicate a problem.
"""
uncovered_paths(pp, pp_sim) = setdiff(keys(pp_sim), keys(pp))
