######################################################################
# argument checking
######################################################################

"Check if argument is a probability vector."
isprobvec{T}(v::AbstractVector{T}) = all(x->x ≥ 0, v) && sum(v) ≈ one(T)

"Check if argument is a probability (stochastic) matrix."
isprobmat(m::AbstractMatrix) = all(i->isprobvec(view(m, i, :)), indices(m, 1))

"Probabilities given are logarithms and valid as distributions."
typealias arg_log Val{:arg_log}
typealias ArgLog Type{arg_log}

"Probabilities given are valid as distributions."
typealias arg_checked Val{:arg_checked}
typealias ArgChecked Type{arg_checked}

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
