"""
Parameters to initialize a Hidden Markov Model: initial probabilibites
`π`, and observation probabilities `b`.

Transformed before storage for faster calculation.
"""
immutable HMMinit
    log_π::AbstractVector       # initial probabilities
    log_b::AbstractMatrix       # observations
    function HMMinit(log_π, log_b, ::ArgLog)
        (N,M) = size(log_b)
        length(log_π) == N ||
            error(DimensionMismatch("non-conformable dimensions"))
        new(log_π, log_b)
    end
end

HMMinit(π, b, ::ArgChecked) = HMMinit(log.(π), log.(b), arg_log)

function HMMinit(π, b)
    isprobvec(π) || error(ArgumentError("π is not a probability vector"))
    isprobmat(b) || error(ArgumentError("b is not a probability matrix"))
    HMMinit(π, b, arg_checked)
end

"""
Parameters updating a Hidden Markov Model: state transition
probabilities `a`, and observation probabilities `b`.

Transformed before storage for faster calculation.
"""
immutable HMMstep
    log_a::AbstractMatrix       # evolution of state
    log_b::AbstractMatrix       # observations
    function HMMstep(log_a, log_b, ::ArgLog)
        (N,M) = size(log_b)
        LinAlg.checksquare(log_a) == N ||
            error(DimensionMismatch("non-conformable dimensions"))
        new(log_a, log_b)
    end
end

HMMstep(a, b, ::ArgChecked) = HMMstep(log.(a), log.(b), arg_log)

function HMMstep(a, b)
    isprobmat(a) || error(ArgumentError("a is not a probability vector"))
    isprobmat(b) || error(ArgumentError("b is not a probability matrix"))
    HMMstep(a, b, arg_checked)
end

"""
Initialize a HMM with parameters in `c`, given first `observation`.
"""
hmm_init(c::HMMinit, observation::Int) = c.log_π + c.log_b[:, observation]

"""
Internal function for HMM updating. Skips calculation for
zero-probability events.
"""
function _hmm_forward(log_α, log_a, log_b)
    log_b == -Inf ? log_b : logsumexp(log_α + log_a) + log_b
end

"""
Update a HMM with parameters in `c`, given `observation` and the
current log likelihood `α`.
"""
function hmm_update(log_α::AbstractVector, c::HMMstep, observation::Int)
    [_hmm_forward(log_α, c.log_a[:, j], log_b_j)
     for (j,log_b_j) in enumerate(c.log_b[:, observation])]
end

"""
Log likelihood of all observations so far.
"""
hmm_log_likelihood(log_α) = logsumexp(log_α)
