module HiddenMarkovChains

using StatsFuns
using DataStructures
using Distributions
using Parameters

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
include("steadystate.jl")
include("hmc.jl")
include("simulation.jl")
include("pathprob.jl")
include("likelihood.jl")


end # module
