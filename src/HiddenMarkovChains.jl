module HiddenMarkovChains

using StatsFuns
using DataStructures
using Distributions

export
    # this file
    HiddenMarkovChain,
    HMC_Observation_Counts,
    # utilities
    enforce_rowsums!,
    # steady state
    steady_state,
    # hmc
    # path probabilities
    path_probabilities,
    # simulation
    simulate_path,
    simulate_paths,
    simulate_path_counts,
    simulate_path_probabilities,
    # likelihood
    path_loglikelihood

include("utilities.jl")

"""
Hidden Markov Chain, with

- initial probabilities `π`, a vector of state transition matrices
- `P`, a vector of observation matrices `Q`.

`Q` determines the length of the chain, and needs to have at least one
more element than `P`. Extra elements of the latter are ignored. For
time-homogenous chains, see Repeated.
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

"Counting observed sequences from HMCs."
typealias HMCPathDict{T} Dict{Vector{Int},T}

include("steadystate.jl")
include("simulation.jl")
include("pathprob.jl")
include("likelihood.jl")

end # module
