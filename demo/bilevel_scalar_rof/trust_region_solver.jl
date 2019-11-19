using AbstractOperators
using LinearAlgebra
using Optim

include("gradient_solver.jl")

@enum TRModels begin
    linear = 1
    sr1 = 2 
    bfgs = 3
end

function trust_region_subproblem(lambda,g,H,radius)
    f_univariate(x) = H*x^2 + g*x
    if lambda >= radius
        res = Optim.optimize(f_univariate, -radius, radius)
    else
        res = Optim.optimize(f_univariate, -lambda, radius)
    end
    return Optim.minimizer(res)
end

function update_bfgs_approximation(grad,grad_prev,lambda,lambda_prev,B)
    s = lambda-lambda_prev
    if abs(s) > 0
        Bs = B*s
        y = grad-grad_prev
        if s*y >= 0
            B = B - Bs.^2/(s'*Bs)+y.^2/(s'*y)
        else
            # B = B - Bs.^2/(s'*Bs)+y.^2/(s'*y)
            # B = -B
            print(" * ")
        end
    # else
    #     println("no change in lambda -- skipping bfgs update")
    end
    return B
end

function update_sr1_approximation(grad,grad_prev,lambda,lambda_prev,B)
    s = lambda-lambda_prev
    if abs(s) > 0
        Bs = B*s
        y = grad-grad_prev
        if norm(y) > 1e-3
            u = y-Bs
            if abs(s'*u) > 1e-9 * norm(s)*norm(u)
                B = B + (u.^2/(u'*s))
            else
                print(" * ")
            end
        end
    # else
    #     println("no change in lambda -- skipping bfgs update")
    end
    return B
end


function trust_region_solver(lower_level_solver::Function,upper_level_cost::Function,lambda_0,α,f,z,radius,tol,model::Integer)
    
    # Variable Initialization
    eta_1 = 0.1
    eta_2 = 0.9
    beta_1 = 0.5
    beta_2 = 1.5

    lambda = lambda_0
    lambda_prev = lambda_0
    K = Variation(size(f))
    u = Variable(size(K,1)...)
    nabla = gradient_matrix(size(f))
    it = 1

    # Second order information
    if model == 1
        H_k = 0
    else
        H_k = 0.1
        u_0 = lower_level_solver(u,f,lambda_prev,K)
        g_prev = gradient_solver(u_0,z,lambda_prev,α,K,nabla)
    end

    while radius > tol
        print("TR Iteration $it: \t")
        u_k = lower_level_solver(u,f,lambda,K)
        g_k = gradient_solver(u_k,z,lambda,α,K,nabla)
        if norm(g_k) < tol
            break
        end

        if model == 2
            H_k = update_sr1_approximation(g_k,g_prev,lambda,lambda_prev,H_k)
        elseif model == 3
            H_k = update_bfgs_approximation(g_k,g_prev,lambda,lambda_prev,H_k)
        end

        # Solve trust region subproblem
        s_k = trust_region_subproblem(lambda,g_k,H_k,radius)

        # Quality indicator calculation
        cost = upper_level_cost(u_k,z,lambda,α)
        u_k_ = lower_level_solver(u,f,lambda+s_k,K)
        cost_ = upper_level_cost(u_k_,z,lambda+s_k,α)
        ared_k = cost-cost_
        pred_k = -g_k'*s_k - 0.5*s_k'*H_k*s_k
        #println("$cost, $cost_, $ared_k, $pred_k")
        rho_k = ared_k/pred_k

        print("rho_k = $(round(rho_k,digits=3)), radius = $(round(radius,digits=4)), lambda = $(round(lambda,digits=4)), g_k = $(round(g_k,digits=3)), s_k = $(round(s_k,digits=3)), H_k = $(round(H_k,digits=2))\n")
        
        # Update
        if rho_k > eta_1
            lambda = lambda+s_k
        end

        if rho_k > eta_2
            radius = radius * beta_2
        elseif rho_k <= eta_1
            radius = radius * beta_1
        end

        # Save previous step
        if lambda-lambda_prev > 0
            lambda_prev = lambda
            g_prev = g_k
        end

        
        it += 1

    end

    return lambda
end

function trust_region_solver(lower_level_solver::Function,upper_level_cost::Function,lambda_0,α,f,z,radius)
    tol = 1e-3
    trust_region_solver(lower_level_solver,upper_level_cost,lambda_0,α,f,z,radius,tol)
end