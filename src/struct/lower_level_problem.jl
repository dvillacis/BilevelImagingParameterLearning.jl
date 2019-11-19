using StructuredOptimization
using AbstractOperators

struct lower_level_problem
    u::Variable
    f::AbstractArray
    λ::AbstractArray
    K::Variation
    solve::Function
end



