
export ScalarROFAdjointOperator

import Base: size

struct ScalarROFAdjointOperator{T,M<:Float64} <: LinearOperator
	u::AbstractArray{M}
    K::Variation
    Ku::AbstractArray{M}
    nKu::AbstractArray{M}
end

# Constructor
###standard constructor Operator{N}(DomainType::Type, DomainDim::NTuple{N,Int})
function ScalarROFAdjointOperator(DomainType::Type, u::AbstractArray{M}, K::Variation, Ku::AbstractArray{M},nKu::AbstractArray{M}) where {N,M}
	ScalarROFAdjointOperator{DomainType,M}(u,K,Ku,nKu)
end
###

function ScalarROFAdjointOperator(u::AbstractArray{M}) where {M}
	K = Variation(size(u))
	Ku = K*u
	nKu = sqrt.(sum(Ku.^2, dims=2))
    ScalarROFAdjointOperator(Float64,u,K,Ku,nKu)
end

# Mappings
function mul!(y::AbstractArray{T,N}, L::ScalarROFAdjointOperator{T,M}, b::AbstractArray{T,D}) where {T,N,M,D}
	Kp = L.K*b
	a1 = Kp./L.nKu
	a2 = Kp*L.Ku'
	#a3 = a2./(L.nKu.^3)
	# a4 = L.Ku.*vcat(a3...)
	y = a1-a2
end

# Properties
domainType(L::ScalarROFAdjointOperator{T, M}) where {T, M} = T
codomainType(L::ScalarROFAdjointOperator{T, M}) where {T, M} = T

size(L::ScalarROFAdjointOperator) = (size(L.Ku),size(L.u))

fun_name(L::ScalarROFAdjointOperator)  = "H"