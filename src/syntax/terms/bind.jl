#bind.jl - Overload of the native norm function to support scale dependent parameter norms

# Norms
import LinearAlgebra: norm
export norm

# Weighted Norm
# Mixed Norm
function norm(ex::AbstractExpression, lambda::AbstractArray, p1::Int, p2::Int, dim::Int = 1 )
	if p1 == 2 && p2 == 1
		f = NormWL21(lambda,dim)
	else
		error("function not implemented")
	end
	return Term(f, ex)
end