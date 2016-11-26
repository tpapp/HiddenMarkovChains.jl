module HiddenMarkovChains

export
    steady_state,
    psp_init,
    psp_observe,
    psp_observe_end,
    psp_forward,
    path_probabilities

"Check if argument is a probability vector."
isprobvec{T}(v::AbstractVector{T}) = all(x->x ≥ 0, v) && sum(v) ≈ one(T)

"Check if argument is a probability (stochastic) matrix."
isprobmat(m::AbstractMatrix) = all(i->isprobvec(view(m, i, :)), indices(m, 1))

######################################################################
# steady state
######################################################################

"A vector from the (left) null space of A."
nullspace1T(A::AbstractMatrix) = qrfact(A)[:Q][:, end]

function nullspace1T(A::SparseMatrixCSC)
    n = size(A, 1)
    e = zeros(n)
    e[n] = 1.0
    q = SparseArrays.SPQR.qmult(SparseArrays.SPQR.QX, qrfact(A),
                                SparseArrays.SPQR.Dense(e))
    convert(Vector, q)
end

"The steady state distribution of a Markov chain with transition matrix P.

Uses the QR decomposition, see Golub and Meyer (1986)."
steady_state(P) = normalize(nullspace1T(P-I), 1)

######################################################################
# lightweight structure for path, state pairs
######################################################################
typealias State Int64
typealias Path Array{State, 1}

typealias PathState Tuple{Path,State}

PathState(path::Path,state::State) = (path, state)
PathState(state::State) = (Array(State,0), state)

path(path_state::PathState) = path_state[1]
state(path_state::PathState) = path_state[2]

add_observation(path::Path, observation::State) = vcat(path, observation)

function add_observation(path_state::PathState, observation::State)
    PathState(add_observation(path(path_state), observation), state(path_state))
end

function replace_state(path_state::PathState, state::State)
    PathState(path(path_state), state)
end

typealias PathStateProb{T <: Real} Dict{PathState, T}

######################################################################
# iterate over nonzeros in columns
######################################################################
function column_nz(M::AbstractMatrix, col::Integer)
    enumerate(view(M, :, col))
end

function column_nz(M::SparseMatrixCSC, col::Integer)
    nzr = nzrange(M, col)
    zip(view(rowvals(M), nzr), view(nonzeros(M), nzr))
end

######################################################################
# probability distribution of paths
######################################################################

"When `prob` > 0, add it to `key` in `dict`."
function add_prob!{T, S <: Real}(dict::Dict{T, S}, key::T, prob)
    if prob > 0
        dict[key] = get(dict, key, zero(S)) + prob
    end
    nothing
end

"Return a `PathStateProb` object that corresponds to `π`."
function psp_init{T}(π::AbstractVector{T})
    psp = PathStateProb{T}()
    for (state, prob) in enumerate(π)
        if prob > 0
            psp[PathState(state)] = prob
        end
    end
    psp
end

"Extend the current ``path,state -> probability`` records by adding
observations to the `path`s based on `state`, with transition probabilities 
`Q[state, observation]`.

Call after `psp_init` or `psp_forward`. Returns new `PathStateProb`
object, does not modify arguments."
function psp_observe{T}(Q::AbstractMatrix, path_state_prob::PathStateProb{T})
    QT = Q'
    next = PathStateProb{T}()
    for (path_state, prob) in path_state_prob
        for (obs, trans_prob) in column_nz(QT, state(path_state))
            next_prob = prob*trans_prob
            if next_prob > 0
                next[add_observation(path_state, obs)] = next_prob
            end
        end
    end
    next
end

"Similar to `psp_observe`, but discard the states. Useful for the last
time period.

Returns new `PathStateProb` object, does not modify arguments."
function psp_observe_end{T}(Q::AbstractMatrix, path_state_prob::PathStateProb{T})
    QT = Q'
    next = Dict{Path,T}()
    for (path_state, prob) in path_state_prob
        for (obs, trans_prob) in column_nz(QT, state(path_state))
            add_prob!(next, add_observation(path(path_state), obs),
                      prob*trans_prob)
        end
    end
    next
end

"Extend the current ``path,state -> probability`` records by updating states according to the transition probability matrix `P`.

Returns new `PathStateProb` object, does not modify arguments."
function psp_forward{T}(P::AbstractMatrix, path_state_prob::PathStateProb{T})
    PT = P'
    next = PathStateProb{T}()
    for (path_state, prob) in path_state_prob
        for (next_state, trans_prob) in column_nz(PT, state(path_state))
            add_prob!(next, replace_state(path_state, next_state), prob*trans_prob)
        end
    end
    next
end

"Return a dictionary that maps observed paths to their probabilities,
for a hidden Markov chain with
- transition matrix `P` for the hidden state,
- transition matrix `Q` for observations,
- initial state distribution `p`,
- running for `T` time periods.

`P` and `Q` can be sparse or dense and are handled accordingly.

Paths with zero probability are not recorded."
function path_probabilities(P::AbstractMatrix, Q::AbstractMatrix,
                            path_state_prob::PathStateProb, T::Int)
    @assert T >= 1
    psp = path_state_prob
    for t in 1:T
        if t == T
            psp = psp_observe_end(Q, psp)
        else
            psp = psp_observe(Q, psp)
            psp = psp_forward(P, psp)
        end
    end
    psp
end

path_probabilities(P, Q, p, T) = path_probabilities(P, Q, psp_init(p), T)

end # module
