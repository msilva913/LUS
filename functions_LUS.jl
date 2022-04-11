using Plots
using LaTeXStrings, KernelDensity
using Parameters, CSV, StatsBase, Statistics, Random
using NLsolve, Dierckx, Distributions, ArgParse
using LinearAlgebra, QuadGK, Roots, Optim

@with_kw mutable struct ParaCalib2


    δ::Float64 = 0.03045
    zbar::Float64 = 1.0
    w0::Float64 = 0.7186
    w1::Float64 = 0.97
    ϕ::Float64 = 0.04702
    ρ::Float64 = 0.002466
    α::Float64 = 0.004449
    B::Float64 = 16.645
    σ::Float64 = 0.2
    k::Float64 = 4.413
    A::Float64 = 0.5631
    ξ::Float64 = 0.5
    Ag::Float64 = 4.529
    λ::Float64 = 0.8
    γ::Float64 = 1.0  # functions assume log utility throughout, caution if change this

end

### Basic functions

function rfunc(J::Float64, n::Float64, para::ParaCalib2)

    @unpack α, λ, B, Ag, ρ, γ, σ, Ag = para
    
    L = min(B, n*J+Ag)
     x = α*(1-λ)*(B/L-1.0)   #  x=0 if lam =1 or L=B 
    return (ρ-x)/(1+x)

end

function zfunc(J::Float64, n::Float64, para::ParaCalib2)

    @unpack zbar, σ, α, λ, B, Ag = para

    L = min(B, n*J+Ag)
    return zbar + (σ/(1+σ))*(α/n)*(λ*B+(1-λ)*L)

end

function wfunc(J::Float64, n::Float64, Je::Float64, ne::Float64, FixFlag::Int64, para::ParaCalib2)

    @unpack A, k, ξ, ϕ, w0, ρ, δ, w1 = para

    β = 1/(1+ρ)

    θ = (A*J/(k*(1+rfunc(J,n,para))))^(1/ξ)
    q = A*θ^(-ξ)
    rp = rfunc(Je,ne,para)

    out = (1-ϕ)*w0 +ϕ*(zfunc(J,n,para)+β*(1+rp)*k*θ) + ϕ*(1-δ)*k*(1-β*(1+rp))/q
    #out = (1-ϕ)*w0 +ϕ*(zfunc(J,n,para)+k*θ)

    rout = FixFlag*w1+(1-FixFlag)*out

    return rout

end




function Jfunc(n::Float64, Je::Float64, ne::Float64, FixFlag::Int64,para::ParaCalib2)

    @unpack zbar, σ, w1, δ, B, Ag, α, λ, ρ = para

    function obj(x::Float64)
        L = min(B, n*x+Ag)
        #zbar+(σ/(1+σ))*(ys(x,n,para))^(1+σ) - w1 +(1-δ)*Je/(1+rfunc(Je, ne, para))-x
        
        return zbar+(σ/(1+σ))*(α/n)*(λ*B+(1-λ)*L) - wfunc(x,n,Je,ne,FixFlag,para) +(1-δ)*Je/(1+rfunc(Je, ne, para))-x

    end

    sol = fzero(x -> obj(x),1.0)
    return sol
end

function Jfunc(n::Float64, Je::Float64, ne::Float64, FixFlag::Int64,para::ParaCalib2, r_fixed::Float64)

    @unpack zbar, σ, w1, δ, B, Ag, α, λ, ρ = para

    function obj(x::Float64)
        L = min(B, n*x+Ag)
        #zbar+(σ/(1+σ))*(ys(x,n,para))^(1+σ) - w1 +(1-δ)*Je/(1+rfunc(Je, ne, para))-x
        
        return zbar+(σ/(1+σ))*(α/n)*(λ*B+(1-λ)*L) - wfunc(x,n,Je,ne,FixFlag,para) +(1-δ)*Je/(1+r_fixed)-x

    end

    sol = fzero(x -> obj(x),1.0)
    return sol
end

function nfunc(n::Float64, Je::Float64, para::ParaCalib2)

    @unpack δ, A, k, ξ = para

    function obj(x::Float64)

       
       return (1-δ)*n+A^(1/ξ)*(Je/(k*(1+rfunc(Je, x, para))))^((1-ξ)/ξ)*(1-n)-x
    end

    sol = fzero(x -> obj(x), 0.9)

    return sol

end

# Functions to compute the countour maps

function Jnull(J_lims::Array{Float64,1}, steps::Int64, FixFlag::Int64,para::ParaCalib2)


    function obj(x::Float64, J::Float64)

        return Jfunc(x, J, x, FixFlag, para)-J

    end

    J_data = range(J_lims[1], stop = J_lims[2], length = steps)|>collect

    sol = map(J -> fzero(x-> obj(x, J), .001), J_data)

    return sol

end

function revJnull(n_lims::Array{Float64,1}, steps::Int64, FixFlag::Int64, para::ParaCalib2)

    function obj(x::Float64, n::Float64)

        return Jfunc(n,x,n,FixFlag, para)-x

    end

    n_data = range(n_lims[1], stop = n_lims[2], length = steps)|>collect

    sol = map(n -> fzero(x->obj(x,n), .001), n_data)

    return sol

end


function nnull(J_lims::Array{Float64,1}, steps::Int64, para::ParaCalib2)

    function obj(x::Float64, J::Float64)

        return nfunc(x, J, para)-x
    end

    J_data = range(J_lims[1], stop = J_lims[2], length = steps)|>collect

    sol = map(J -> fzero(x-> obj(x, J), 0.9), J_data)

    return sol

end


function fp(FixFlag::Int64, para::ParaCalib2)

    function obj(x::Array{Float64,1})

        obj1 = Jfunc(x[2], x[1], x[2], FixFlag, para)-x[1]
        obj2 = nfunc(x[2],x[1],para)-x[2]
        return [obj1 obj2]
    end        

    # check high steady state
    sol1 = nlsolve(obj, [4.1;.5])
    # check low steady state
    sol2 =  nlsolve(obj,[4.1; 0.05])  
        
    return [sol1.zero  sol2.zero]
end


function fp(FixFlag::Int64, para::ParaCalib2, r_fixed::Float64)

    function obj(x::Array{Float64,1})

        obj1 = Jfunc(x[2], x[1], x[2], FixFlag, para, r_fixed)-x[1]
        obj2 = nfunc(x[2],x[1],para)-x[2]
        return [obj1 obj2]
    end        

    # check high steady state
    sol1 = nlsolve(obj, [4.1;.5])
    # check low steady state
    sol2 =  nlsolve(obj,[4.1; 0.05])  
        
    return [sol1.zero  sol2.zero]
end

function steady_state(FixFlag::Int64, para::ParaCalib2)
    " Return steady-state values as named tuple "
    @unpack A, k, ξ, ρ, Ag = para
    ss = fp(FixFlag, para)
    J, n = ss[:, 1]
    M = n*J
    u = 1 - n
    r = rfunc(J, n, para)
    z = zfunc(J, n, para)
    θ = (A*J/(k*(1+r)))^(1/ξ)
    w = wfunc(J, n, J, n, FixFlag, para)
    spread = ρ - r
    debt_gdp = Ag/(n*z)
    # output as named tuple 
    out = (n=n, J=J, M=M, u=u, θ=θ, w=w, z=z, r=r, spread=spread, x=debt_gdp)
    return out
end


function ParaTrans(FixFlag::Int64, x::Float64, para::ParaCalib2)
    """ 
    Struct in terms of debt_GDP rather than Ag
    Map debt-to-GDP x to Ag
    """
    para_new = deepcopy(para)
    function output_loss(y)
        y = sqrt(y^2)
        Ag = x*y
        # modify instance
        para_new.Ag = Ag
        # calculate steady state
        steady = steady_state(FixFlag, para)
        return 100*(steady.n*steady.z-y)/y, para_new
    end

    # Find root
    sol = fzero(y -> output_loss(y)[1], 0.9)
    res, para = output_loss(sol)
    return para
end
