"Keep track of `(state,path) => probability`."
typealias _PP{T} Dict{Tuple{Int,Vector{Int}}, T}

"Initialize dictionary."
_pp_init(π) = Dict((i,Int[]) => prob for (i,prob) in enumerate(π) if prob > 0)

"Add next observation, using observation probabilities `Q`."
function _pp_observe{T,S}(pp::_PP{T}, Q::AbstractMatrix{S})
    QT = Q'
    next_pp = _PP{promote_type(T,S)}()
    for ((state, path), prob) in pp
        for (next_observation, trans_prob) in column_nz(QT, state)
            next_prob = prob*trans_prob
            if next_prob > 0
                next_pp[(state, vcat(path, next_observation))] = next_prob
            end
        end
    end
    next_pp
end

"When `x > 0`, add it to `key` in `dict`."
function _add_if_positive!{T, S <: Real}(dict::Dict{T, S}, key::T, x)
    if x > 0
        dict[key] = get(dict, key, zero(S)) + x
    end
end


"Update states according to the transition probability matrix `P`."
function _pp_forward{T,S}(pp::_PP{T}, P::AbstractMatrix{S})
    PT = P'
    next_pp = _PP{promote_type(T,S)}()
    for ((state, path), prob) in pp
        for (next_state, trans_prob) in column_nz(PT, state)
            _add_if_positive!(next_pp, (next_state, path), prob*trans_prob)
        end
    end
    next_pp
end


"""
Similar to `_pp_observe`, but sum over the states. Useful for the last
time period.
"""
function _pp_end{T,S}(pp::_PP{T}, Q::AbstractMatrix{S})
    QT = Q'
    next_pp = HMCPathDict{promote_type(T,S)}()
    for ((state, path), prob) in pp
        for (observation, trans_prob) in column_nz(QT, state)
            _add_if_positive!(next_pp, vcat(path, observation), prob*trans_prob)
        end
    end
    next_pp
end


"Return a HMCPathDict that maps observed paths to their probabilities,
for a hidden Markov chain `h`. Paths with zero probability are not
recorded."
function path_probabilities(h::HiddenMarkovChain)
    # initialize with starting probabilities
    pp = _pp_init(h.π)
    T = length(h)
    for t in 1:T
        if t == T
            pp = _pp_end(pp, h.Q[t])
        else
            pp = _pp_observe(pp, h.Q[t])
            pp = _pp_forward(pp, h.P[t])
        end
    end
    pp
end
