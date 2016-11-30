"""
Hidden Markov Chain, transformed to log probabilities for faster
calculations.
"""
immutable HiddenMarkovChain_log
    log_π::AbstractVector
    log_P::AbstractVector
    log_Q::AbstractVector
    function HiddenMarkovChain_log(log_π, log_P, log_Q)
        _check_HMC_PQ(log_P, log_Q)
        new(log_π, log_P, log_Q)
    end
end

Base.length(h::HiddenMarkovChain_log) = length(h.log_Q)

HiddenMarkovChain_log(h::HiddenMarkovChain) =
    HiddenMarkovChain_log(log(h.π), map(log, h.P), map(log, h.Q))

"""
Initialize log-likelihood of being in each state, given initial
distribution and first observation.
"""
_ll_init(log_π, log_Q, observation) = log_π + log_Q[:, observation]

"""
Internal function for LL updating. Skips calculation for
zero-probability events.
"""
function _ll_step_shortcut(log_α, log_p, log_q)
    log_q == -Inf ? log_q : logsumexp(log_α + log_p) + log_q
end

"""
Update log-likelihood of being in each state, given log transition and
observation probabilities, and an observation.
"""
function _ll_step(log_α, log_P, log_Q, observation::Int)
    [_ll_step_shortcut(log_α, log_P[:, j], log_q_j)
     for (j,log_q_j) in enumerate(log_Q[:, observation])]
end

"Log likelihood of `path` under HMC `h`."
function path_loglikelihood(h::HiddenMarkovChain_log, path::AbstractVector)
    T = length(h)
    @assert T == length(path)
    log_α = _ll_init(h.log_π, h.log_Q[1], path[1])
    for t in 2:T
        log_α = _ll_step(log_α, h.log_P[t-1], h.log_Q[t], path[t])
    end
    logsumexp(log_α)
end

"""
Log likelihood for a collection of `path => count` pairs for a
HMC.
"""
function path_loglikelihood(h::HiddenMarkovChain_log,
                            path_counts::HMCPathDict)
    ll(h, path_count) = path_count[2] == 0 ? 0 :
        path_count[2]*path_loglikelihood(h, path_count.first)
    sum(ll(h, path_count) for path_count in path_counts)
end

path_loglikelihood(h::HiddenMarkovChain, x) = path_loglikelihood(HiddenMarkovChain_log(h), x)
