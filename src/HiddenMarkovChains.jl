module HiddenMarkovChains

using StatsFuns

export
# steady state
steady_state,
# path state probabilities
psp_init,
psp_observe,
psp_observe_end,
psp_forward,
path_probabilities,
# likelihood
HMMinit,
HMMstep,
hmm_init,
hmm_update,
hmm_log_likelihood

include("utilities.jl")
include("steadystate.jl")
include("pathstate.jl")
include("likelihood.jl")

end # module
