using AbstractOperators
using LinearAlgebra
using Optim

include("gradient_solver.jl")

function trust_region_subproblem(lambda,g,H,radius)
    f_univariate(x) = H*x^2 + g*x
    res = Optim.optimize(f_univariate, -min(lambda,radius), radius)
    return Optim.minimizer(res)
end

function trust_region_solver(lower_level_solver::Function,upper_level_cost::Function,lambda_0,f,z,radius,tol)
    
    alpha = 0.1
    
    # Variable Initialization
    eta_1 = 0.1
    eta_2 = 0.9
    beta_1 = 0.5
    beta_2 = 1.5

    lambda = lambda_0
    K = Variation(size(f))
    u = Variable(size(K,1)...)
    nabla = gradient_matrix(size(f))
    it = 1

    while radius > tol
        print("TR Iteration $it: ")
        u_k = lower_level_solver(u,f,lambda,K)
        g_k = gradient_solver(u_k,z,lambda,alpha,K,nabla)
        s_k = trust_region_subproblem(lambda,g_k,0,radius)

        # Quality indicator calculation
        cost = upper_level_cost(u_k,z,lambda,alpha)
        u_k_ = lower_level_solver(u,f,lambda+s_k,K)
        cost_ = upper_level_cost(u_k_,z,lambda+s_k,alpha)
        ared_k = cost-cost_
        q_k = cost + g_k'*s_k + 0.5*norm(s_k)^2
        pred_k = cost-q_k
        rho_k = ared_k/pred_k
        
        # Update
        if rho_k > eta_1
            lambda = lambda+s_k
        end

        if rho_k > eta_2
            radius = radius * beta_2
        elseif rho_k <= eta_1
            radius = radius * beta_1
        end

        print("rho_k = $rho_k, radius = $radius, lambda = $lambda\n")
        it += 1

    end

    return lambda
end

function trust_region_solver(lower_level_solver::Function,upper_level_cost::Function,lambda_0,f,z,radius)
    tol = 1e-3
    trust_region_solver(lower_level_solver,upper_level_cost,lambda_0,f,z,radius,tol)
end