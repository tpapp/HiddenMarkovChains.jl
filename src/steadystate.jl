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
function steady_state(P)
    isprobmat(P) ||
        error(ArgumentError("argument needs to be a stochastic matrix"))
    π = nullspace1T(P-I)
    # NOTE not using `normalize(π,1)` because of
    # https://github.com/JuliaDiff/ForwardDiff.jl/issues/175
    π ./ sum(π)
end
