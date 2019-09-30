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

end