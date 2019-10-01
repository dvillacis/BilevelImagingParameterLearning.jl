module BilevelImagingParameterLearning

using AbstractOperators
using ProximalOperators
using ProximalAlgorithms

# imports
import StructuredOptimization: Term, AbstractExpression
import ProximalOperators: RealOrComplex, ProximableFunction

# Norms
include("norms/NormWL21.jl")

# Bindings
include("syntax/terms/bind.jl")

# Aux
function Base.show(io::IO, f::ProximableFunction)
    println(io, "description : ", fun_name(f))
    println(io, "domain      : ", fun_dom(f))
    println(io, "expression  : ", fun_expr(f))
    print(  io, "parameters  : ", fun_params(f))
end

end