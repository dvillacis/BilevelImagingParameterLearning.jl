#bind.jl - Overload of the native norm function to support scale dependent parameter norms

# Norms
import LinearAlgebra: norm
export norm

# Weighted norms

# Weighted TV Norm

function norm(ex::AbstractExpression, lambda::AbstractVector, p1::Int, p2::Int, dim::Int = 1 )
	if p1 == 2 && p2 == 1
		f = NormWL21(lambda,dim)
	else
		error("function not implemented")
	end
	return Term(f, ex)
end

# Weighted Least square terms

using ProximalOperators
export weighted_ls

"""
	ls(x::AbstractExpression,lambda::AbstractVector)
Returns the weighted squared norm (least squares) of `x`:
```math
f(x) = \\tfrac{1}{2}∑_i λ_i x_i^2.
```
"""
weighted_ls(ex,lambda) = Term(SqrNormL2(lambda), ex)
