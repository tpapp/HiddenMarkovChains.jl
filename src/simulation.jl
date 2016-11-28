# this is necessary for sparse vectors etc.
function Distributions.Categorical{T <: Real}(v::AbstractVector{T})
    Categorical(collect(v))
end

"Sampler for a vector of probabilities."
_vector_sampler(v) = sampler(Categorical(v))

"A vector of sampler from rows of a stochastic matrix."
_matrix_row_samplers(m) = [_vector_sampler(m[row, :]) for row in indices(m, 1)]

"""
Hidden Markov Chain transformed for simulating.
"""
immutable HiddenMarkovChain_sampler
    π
    P
    Q
    function HiddenMarkovChain_sampler(π, P, Q)
        _check_HMC_PQ(P, Q)
        new(π, P, Q)
    end
end

Base.length(h::HiddenMarkovChain_sampler) = length(h.Q)

function HiddenMarkovChain_sampler(h::HiddenMarkovChain)
    HiddenMarkovChain_sampler(_vector_sampler(h.π),
                              map(_matrix_row_samplers, h.P),
                              map(_matrix_row_samplers, h.Q))
end

"""
Simulate a path of observations from a Hidden Markov Chain.
"""
function simulate_observations(h::HiddenMarkovChain_sampler)
    T = length(h)
    state = rand(h.π)
    observations = Array(Int64, T)
    for t in 1:T
        observations[t] = rand(h.Q[t][state])
        if t != T
            state = rand(h.P[t][state])
        end
    end
    observations
end

simulate_observations(h) = simulate_observations(HiddenMarkovChain_sampler(h))

"""
Simulate `K` Hidden Markov Chains with the given parameters `π`, `P`,
`Q`, and return the count of observations as an Accumulator.
"""
function simulate_observation_counts(h::HiddenMarkovChain_sampler, K::Int)
    c = counter(Vector{Int})
    for _ in 1:K
        push!(c, simulate_observations(h))
    end
    c
end

simulate_observation_counts(h, K) =
    simulate_observation_counts(HiddenMarkovChain_sampler(h), K)

function simulate_observation_probabilities(h, K::Int)
    Dict(key => count/K for (key,count) in
         simulate_observation_counts(h, K))
end
