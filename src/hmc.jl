"""
Check that P has sufficient length. The length of the HMC is defined
by the length of Q.
"""
_check_HMC_PQ(P, Q) = @assert length(P)+1 >= length(Q)

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
