"Check if argument is a probability vector."
isprobvec{T}(v::AbstractVector{T}) = all(x->x ≥ 0, v) && sum(v) ≈ one(T)

"Check if argument is a probability (stochastic) matrix."
isprobmat(m::AbstractMatrix) = all(i->isprobvec(view(m, i, :)), indices(m, 1))


"Iteration over (a superset) of nonzeros in a matrix column."
column_nz(M::AbstractMatrix, col::Integer) = enumerate(view(M, :, col))

function column_nz(M::SparseMatrixCSC, col::Integer)
    nzr = nzrange(M, col)
    zip(view(rowvals(M), nzr), view(nonzeros(M), nzr))
end

"""
Check that P has sufficient length. The length of the HMC is defined
by the length of Q.
"""
_check_HMC_PQ(P, Q) = @assert length(P)+1 >= length(Q)

"""
Repeating the same value `count` times.

The most important function of `map`, since it allows us to apply
functions only once.
"""
immutable Repeated{T,S <: Integer} <: AbstractVector{T}
    value::T
    count::S
end

Base.start(x::Repeated) = 1
Base.next(x::Repeated, state) = (x.value, state+1)
Base.done(x::Repeated, state) = state > x.count

Base.iteratorsize(::Type{Repeated}) = Base.HasLength()
Base.iteratoreltype(::Type{Repeated}) = Base.HasEltype()
Base.eltype{T,S}(::Repeated{T,S}) = T
Base.length(x::Repeated) = x.count
Base.size(x::Repeated) = (length(x), )

Base.getindex(x::Repeated, i) = x.value
Base.endof(x::Repeated) = x.count

Base.show(io::IO, x::Repeated) = invoke(show, Tuple{IO, Any}, io, x)

Base.map(f, x::Repeated) = Repeated(f(x.value), x.count)
