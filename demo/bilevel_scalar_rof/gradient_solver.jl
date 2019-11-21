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

function prodesc(p,q,m,n,nKu)
    p = reshape(p,m*n,2)
    q = reshape(q,m*n,2)
    s = 0
    for i = 1:m*n
        if nKu[i] > 1e-3
            s += p[i,:]'*q[i,:]
        end
    end
    return s
end

#TODO: Seria bueno que K tenga tambien una representacion matricial
function gradient_solver(u,f,z,lambda,alpha,K,nabla)
    m,n = size(u)
    sz = m*n
    K = Variation(size(u))
    Ku = K*u
    nKu = vec(sqrt.(sum(Ku.^2,dims=2))) # Pointwise euclidean norm
    nKu = vcat(nKu,nKu) # Replication of the norm to be the same size as Ku
    act = nKu.<1e-9
    inact = 1 .- act
    Act = spdiagm(0=>act)
    Inact = spdiagm(0=>inact)
    den = Inact*nKu+act
    prodKuKu = outer_product(Ku[:]./(den.^3),Ku[:],m,n)
    A = lambda*sparse(I,sz,sz)
    B = nabla'
    C = -Inact*(prodKuKu-spdiagm(0=> 1 ./ den))*nabla
    D = Act*nabla
    Adj = [A B;D-C Inact+sqrt(eps())*Act]
    Track = [z[:]-u[:];zeros(2*sz,1)]
    #mult,ch = idrs(Adj,Array(Track),tol=1e-2,log=true)
    # println(ch.isconverged)
    # adj = mult[1:sz]
    mult = Adj\Track
    adj = mult[1:sz]
    #return adj
    #Kp = K*reshape(adj,m,n)
    #grad = prodesc(Kp,Ku[:]./den,m,n,nKu) + alpha*lambda
    grad = (u[:]-f[:])'*adj + alpha*lambda
    return grad
end
