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
    Gx = -sparse(collect(1:sz),idx1[:],vec(ones(sz,1)),sz,sz) + sparse(collect(1:sz),idx2[:],vec(ones(sz,1)),sz,sz)

    # u_[i+1,j] - u_[i,j]
    idx1 = idx
    idx2 = idx[vcat(2:m,m),:]
    Gy = -sparse(collect(1:sz),idx1[:],vec(ones(sz,1)),sz,sz) + sparse(1:sz,idx2[:], vec(ones(sz,1)),sz,sz)

    return [Gx;Gy]
end

function gradient_matrix(s::Tuple)
    m = s[1]
    n = s[2]
    return gradient_matrix(m,m)
end

#TODO: Seria bueno que K tenga tambien una representacion matricial
function gradient_solver(u,ut,lambda,K,nabla)
    m,n = size(u)
    sz = m*n
    K = Variation(size(u))
    Ku = K*u
    nKu = vec(sqrt.(sum(Ku.^2,dims=2))) # Pointwise euclidean norm
    nKu2 = vcat(nKu,nKu) # Replication of the norm to be the same size as Ku
    act = nKu2.<1e-3
    inact = 1 .- act
    Act = spdiagm(0=>act)
    Inact = spdiagm(0=>inact)
    den = Inact*nKu2+act
    prodKuKu = outer_product(Ku[:]./(den.^3),Ku[:],m,n)
    A = sparse(I,sz,sz)
    B = nabla'
    C = -Inact*(prodKuKu-spdiagm(0=> 1 ./ den))*nabla
    D = sparse(I,m,n)
    E = Act*nabla
    Adj = [A B;E-C Inact+sqrt(0.1)*Act]
    Track = [u[:]-ut[:];spzeros(2*sz,1)]
    mult,ch = cg(Adj,Track,maxiter=100,tol=1e-2,log=true)
    adj = mult[1:sz]

    beta = 0
    gamma = Ku./hcat(nKu,nKu)
    t = sparse(K'*gamma)
    println(typeof(t[:]))
    println(typeof(adj))
    return t[:]*adj #+ beta*lambda
end
