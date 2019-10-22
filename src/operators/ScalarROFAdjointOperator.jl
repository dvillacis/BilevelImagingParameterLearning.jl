
export ScalarROFAdjointOperator

struct ScalarROFAdjointOperator{T,N} <: LinearOperator
    lambda::Float64
    dim_in::NTuple{N,NTuple{N,Int}}
end

# Constructors
#default constructor
function ScalarROFAdjointOperator(domainType::Type, lambda::Float64, dim_in::NTuple{N,NTuple{N,Int}}) where {N} 
    N == 1 && error("use FiniteDiff instead!")
    K = Variation(dim_in[1])
    ScalarROFAdjointOperator{domainType,N}(lambda,dim_in)
end

#ScalarROFAdjointOperator(domainType::Type, lambda::Float64, dim_in::Vararg{Int}) = ScalarROFAdjointOperator(domainType, lambda, dim_in)
ScalarROFAdjointOperator(lambda::Float64, dim_in::NTuple{N,NTuple{N,Int}}) where {N} = ScalarROFAdjointOperator(Float64, lambda, dim_in)
#ScalarROFAdjointOperator(lambda::Float64, dim_in::Vararg{Int}) = ScalarROFAdjointOperator(lambda, dim_in)
#ScalarROFAdjointOperator(x::AbstractArray, y::AbstractArray, lambda::Float64)  = ScalarROFAdjointOperator(eltype(x), size(x), size(y), lambda)

@generated function mul!(y::AbstractArray{T,2}, A::ScalarROFAdjointOperator{T,N}, b::AbstractArray{T,N}, c::AbstractArray{T,N}) where {T,N}
    return [b - A.lambda * A.K'* c; c - A.K * b]
end

# Properties

domainType(L::ScalarROFAdjointOperator{T,N}) where {T,N} = T
codomainType(L::ScalarROFAdjointOperator{T,N}) where {T,N} = T

size(L::ScalarROFAdjointOperator{T,N}) where {T,N} = (L.dim_in[1], L.dim_in[2])

fun_name(L::BilevelImagingParameterLearning.ScalarROFAdjointOperator)  = "ScalarROFAdjointOperator"