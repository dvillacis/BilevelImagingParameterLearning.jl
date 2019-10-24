export Identity

"""
`Identity([domainType=Float64::Type,] dim_in::Tuple)`
`Identity([domainType=Float64::Type,] dims...)`
Create the identity operator.
```julia
julia> op = Identity(Float64,(4,))
I  ℝ^4 -> ℝ^4
julia> op = Identity(2,3,4)
I  ℝ^(2, 3, 4) -> ℝ^(2, 3, 4)
julia> op*ones(2,3,4) == ones(2,3,4)
true
```
"""
struct Identity{T, N} <: LinearOperator
	dim::NTuple{N, Integer}
end

# Constructors
###standard constructor Operator{N}(DomainType::Type, DomainDim::NTuple{N,Int})
Identity(DomainType::Type, DomainDim::NTuple{N,Int}) where {N} = Identity{DomainType,N}(DomainDim)  
###

Identity(t::Type, dims::Vararg{Integer}) = Identity(t,dims)
Identity(dims::NTuple{N, Integer}) where {N} = Identity(Float64,dims)
Identity(dims::Vararg{Integer}) = Identity(Float64,dims)
Identity(x::A) where {A <: AbstractArray} = Identity(eltype(x), size(x))

# Mappings

mul!(y::AbstractArray{T, N}, L::Identity{T, N}, b::AbstractArray{T, N}) where {T, N} = y .= b
mul!(y::AbstractArray{T, N}, L::AdjointOperator{Identity{T, N}}, b::AbstractArray{T, N}) where {T, N} = mul!(y, L.A, b)

# Properties
diag(L::Identity) = 1.
diag_AcA(L::Identity) = 1.
diag_AAc(L::Identity) = 1.

domainType(L::Identity{T, N}) where {T, N} = T
codomainType(L::Identity{T, N}) where {T, N} = T

size(L::Identity) = (L.dim, L.dim)

fun_name(L::Identity) = "I"

is_Identity(L::Identity) = true
is_diagonal(L::Identity) = true
is_AcA_diagonal(L::Identity) = true
is_AAc_diagonal(L::Identity) = true
is_orthogonal(L::Identity) = true
is_invertible(L::Identity) = true
is_full_row_rank(L::Identity) = true
is_full_column_rank(L::Identity) = true