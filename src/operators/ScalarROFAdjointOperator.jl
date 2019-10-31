
export ScalarROFAdjointOperator

import Base: size

struct ScalarROFAdjointOperator{T,M<:Float64} <: LinearOperator
	u::AbstractArray{M}
    K::Variation
    Ku::AbstractArray{M}
	nKu::AbstractArray{M}
	lambda::Float64
	n::Integer
	m::Integer
end

# Constructor
###standard constructor Operator{N}(DomainType::Type, DomainDim::NTuple{N,Int})
function ScalarROFAdjointOperator(DomainType::Type, u::AbstractArray{M}, K::Variation, Ku::AbstractArray{M},nKu::AbstractArray{M},lambda::Float64,n::Integer,m::Integer) where {N,M}
	ScalarROFAdjointOperator{DomainType,M}(u,K,Ku,nKu,lambda,n,m)
end
###

function ScalarROFAdjointOperator(u::AbstractArray{M},lambda::Float64) where {M}
	K = Variation(size(u))
	Ku = K*u
	nKu = sqrt.(sum(Ku.^2, dims=2))
	(n,m) = size(u)
    ScalarROFAdjointOperator(Float64,u,K,Ku,nKu,lambda,n,m)
end

# Mappings
function mul!(y::AbstractArray{T,N}, L::ScalarROFAdjointOperator{T,M}, b::AbstractArray{T,D}) where {T,N,M,D}
	y = b[1] - L.lambda * L.K' * b[2]
end

# Properties
domainType(L::ScalarROFAdjointOperator{T, M}) where {T, M} = T
codomainType(L::ScalarROFAdjointOperator{T, M}) where {T, M} = T

size(L::ScalarROFAdjointOperator) = ((3*L.m*L.n,1),(3*L.m*L.n,1))

fun_name(L::ScalarROFAdjointOperator)  = "H"