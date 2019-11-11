using StructuredOptimization
using Images

function lower_level_solver_rof(lambda,Y,V,U)
    @minimize ls(-V'*U+Y) + conj(lambda*norm(U,2,1,2)) with ForwardBackward(tol = 1e-3, gamma = 1/8, fast = true) 
    return Gray.(-V'*(~U)+Y)
end