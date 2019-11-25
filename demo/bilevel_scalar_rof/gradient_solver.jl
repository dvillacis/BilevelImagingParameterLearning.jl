using AbstractOperators
using SparseArrays
using LinearAlgebra
using IterativeSolvers

function outer_product(p,q,m,n)
    p = reshape(p,m*n,2)
    q = reshape(q,m*n,2)
    a = p[:,1].*q[:,1]
    b = p[:,1].*q[:,2]
    c = p[:,2].*q[:,1]
    d = p[:,2].*q[:,2]
    return [spdiagm(0=>a) spdiagm(0=>b);spdiagm(0=>c) spdiagm(0=>d)]
end

function gradient_matrix(m,n)
    sz = m*n
    idx = reshape(1:sz,m,n)

    # u_[i,j+1] - u_[i,j]
    idx1 = idx
    idx2 = idx[:,vcat(2:n,n)]
    one_vec = vec(ones(sz,1))
    Gx = -sparse(collect(1:sz),idx1[:],one_vec,sz,sz) + sparse(collect(1:sz),idx2[:],one_vec,sz,sz)

    # u_[i+1,j] - u_[i,j]
    idx1 = idx
    idx2 = idx[vcat(2:m,m),:]
    Gy = -sparse(collect(1:sz),idx1[:],one_vec,sz,sz) + sparse(1:sz,idx2[:], one_vec,sz,sz)
    return vcat(Gx,Gy)
end

function gradient_matrix(s::Tuple)
    m = s[1]
    n = s[2]
    return gradient_matrix(m,m)
end

#TODO: Seria bueno que K tenga tambien una representacion matricial
function adjoint_solver_nonreg(u,Ku,nKu,f,z,λ,K,∇)
    m,n = size(u)
    sz = m*n
    L = λ*ones(sz,1)
    L = spdiagm(0=>vec(L))
    act = nKu.<1e-9
    inact = 1 .- act
    Act = spdiagm(0=>act)
    Inact = spdiagm(0=>inact)
    den = Inact*nKu+act
    prodKuKu = outer_product(Ku[:]./(den.^3),Ku[:],m,n)
    A = λ*sparse(I,sz,sz)
    B = ∇'
    C = -Inact*(prodKuKu-spdiagm(0=> 1 ./ den))*∇
    D = Act*∇
    Adj = [A B;D-C Inact+sqrt(eps())*Act]
    Track = [z[:]-u[:];zeros(2*sz,1)]
    mult = Adj\Track
    adj = mult[1:sz]
    return adj
end

function adjoint_solver_reg(u,Ku,nKu,f,z,λ,K,∇)
    m,n = size(u)
    γ = 1000
    sz = m*n
    K = Variation(size(u))

    L = λ*ones(sz,1)
    L = spdiagm(0=>vec(L))
    
    act1 = γ * nKu .- 1
    act = sparse(max.(0,act1[1:sz] .- 1 ./ (2*γ)) .!= 0)
    inact = sparse(min.(0,act1[1:sz] .+ 1 ./ (2*γ)) .!= 0)
    sact = sparse(1 .- act .- inact)
    Act = spdiagm(0=>act)
    Sact = spdiagm(0=>sact)
    # # Inact = spdiagm(0=>inact)
    den = (Act+Sact)*nKu[1:sz]+inact
    
    # Diagonal matrix corresponding to regularization function
    # # den=(Act+Sact)*nKu[1:sz]+inact
    mk=(act+Sact*(1 .- γ/2*(1 .- γ*nKu[1:sz] .+1/(2*γ)).^2))./den
    Dmi=spdiagm(0=>vec(kron(ones(2,1),mk+γ*inact)))

    # Negative term in the derivative
    subst=spdiagm(0=>act+Sact*(1 .- γ/2 * (1 .- γ*nKu[1:sz] .+ 1 ./ (2*γ)).^2))
    subst=kron(I(2),subst)

    H4=outer_product(Ku[:] ./ kron(ones(2,1),den.^2),Ku,m,n)

    prodKuKu=outer_product(Ku[:],Ku[:],m,n)

    sk2=(Sact*γ^2*(1 .- γ*nKu[1:sz] .+ 1 ./ (2*γ)))./(den.^2)
    sk2=spdiagm(0=>vec(kron(ones(2,1),sk2)))

    hess22=Dmi*∇ - kron(I(2),(Act+Sact))*Dmi*H4*∇ + sk2*prodKuKu*∇

    A = λ*sparse(I,sz,sz)
    B = ∇';
    adj=(A+B*hess22+L)\(z[:]-u[:])

    return adj
end

function gradient_solver(u,f,z,λ,α,K,∇,radius,radiusmin)
    m,n = size(u)
    K = Variation(size(u))
    Ku = K*u
    nKu = vec(sqrt.(sum(Ku.^2,dims=2))) # Pointwise euclidean norm
    nKu = vcat(nKu,nKu) # Replication of the norm to be the same size as Ku

    adj = 0
    if radius > radiusmin
        adj = adjoint_solver_nonreg(u,Ku,nKu,f,z,λ,K,∇)
    else
        adj = adjoint_solver_reg(u,Ku,nKu,f,z,λ,K,∇)
    end
    return (u[:]-f[:])'*adj + α * λ
end

function gradient_solver(u,f,z,λ,α,K,∇,radius)
    radiusmin = 0.1
    return gradient_solver(u,f,z,λ,α,K,∇,radius,radiusmin)
end