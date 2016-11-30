module HiddenMarkovChains

using StatsFuns
using DataStructures
using Distributions

export
# steady state
steady_state,
# hmc
HiddenMarkovChain,
# path probabilities
path_probabilities,
# simulation
simulate_observations,
simulate_observation_counts,
simulate_observation_probabilities,
# likelihood
log_likelihood

include("utilities.jl")

"""
Hidden Markov Chain.
"""
immutable HiddenMarkovChain
    π::AbstractVector
    P::AbstractVector
    Q::AbstractVector
    function HiddenMarkovChain(π, P, Q)
        _check_HMC_PQ(P, Q)
        new(π, P, Q)
    end
end

Base.length(x::HiddenMarkovChain) = length(x.Q)

function HiddenMarkovChain(π::AbstractVector, P::AbstractMatrix,
                           Q::AbstractMatrix, T::Integer)
    HiddenMarkovChain(π, Repeated(P, T), Repeated(Q, T))
end

HiddenMarkovChain(P::AbstractMatrix, Q, T::Integer) =
    HiddenMarkovChain(steady_state(P), P, Q, T)

include("steadystate.jl")
include("simulation.jl")
include("pathprob.jl")
include("likelihood.jl")

end # module
