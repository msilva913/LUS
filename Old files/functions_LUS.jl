using Plots
using LaTeXStrings, KernelDensity
using Parameters, CSV, StatsBase, Statistics, Random
using NLsolve, Dierckx, Distributions, ArgParse
using LinearAlgebra, QuadGK, Roots, Optim
using PrettyPrinting

@with_kw mutable struct ParaCalib2

    δ::Float64 = 0.03045
    zbar::Float64 = 1.0
    w0::Float64 = 0.6113
    w1::Float64 = 0.97
    ϕ::Float64 = 0.0033
    ρ::Float64 = 0.00341
    α::Float64 = 0.004446
    B::Float64 = 16.656
    σ::Float64 = 0.2
    k::Float64 = 9.20
    A::Float64 = 0.5631
    ξ::Float64 = 0.5
    Bg::Float64 = 4.529
    λ::Float64 = 0.8
    η::Float64 = 0.4794

    # autocorrelation coefficients
    ρx::Float64 = 0.979
    ρB::Float64 = 0.979
    ρα::Float64 = 0.979
    ρη::Float64 = 0.979

    # standard deviations
    σx::Float64 = 0.007
    σB::Float64 = 0.07
    σα::Float64 = 0.07
    ση::Float64 = 0.07

end

### Basic functions

function rbfunc(J::Float64, n::Float64, para::ParaCalib2)
    # Interest rate on bonds
    @unpack α, λ, B, Bg, ρ, σ, Bg, η   = para
    
    L = min(B, η*n*J+Bg)
     x = α*(1-λ)*(B/L-1.0)   #  x=0 if lam =1 or L=B 
    return (ρ-x)/(1+x)
end

function shock_transform(ss_val, shock)
    return ss_val*shock/(1-ss_val+ss_val*shock)
end

function rbfunc(J::Float64, n::Float64, para::ParaCalib2; B_shock=1.0, α_shock = 1.0, η_shock=1.0)
    # Business cycle version
    # Interest rate on bonds
    @unpack α, λ, B, Bg, ρ, σ, Bg, η   = para
    # Apply shocks
    B = B*B_shock
    α = shock_transform(α, α_shock)
    η = shock_transform(η, η_shock)

    L = min(B, η*n*J+Bg)
     x = α*(1-λ)*(B/L-1.0)   #  x=0 if lam =1 or L=B 
    return (ρ-x)/(1+x)
end

# Wrapper function (useful for broadcasting)
rbfunc(J, n, para, B_shock, α_shock, η_shock) = rbfunc(J, n, para; B_shock, α_shock, η_shock)

function rafunc(J::Float64, n::Float64, para::ParaCalib2)
    # Interest rate on equity
    @unpack α, λ, B, Bg, ρ, σ, Bg, η   = para
    
    L = min(B, η*n*J+Bg)
     x = α*(1-λ)*η*(B/L-1.0)   #  x=0 if lam =1 or L=B 
    return (ρ-x)/(1+x)
end

function rafunc(J::Float64, n::Float64, para::ParaCalib2; B_shock=1.0, α_shock = 1.0, η_shock=1.0)
    # Interest rate on equity
    # Business cycle version
    @unpack α, λ, B, Bg, ρ, σ, Bg, η   = para
     # Apply shocks
     B = B*B_shock
     α = shock_transform(α, α_shock)
     η = shock_transform(η, η_shock)
    
    L = min(B, η*n*J+Bg)
     x = α*(1-λ)*η*(B/L-1.0)   #  x=0 if lam =1 or L=B 
    return (ρ-x)/(1+x)
end

# Wrapper function for broadcasting
rafunc(J, n, para, B_shock, α_shock, η_shock) = rafunc(J, n, para; B_shock, α_shock, η_shock)

function bond_spread(J, n, para)
    @unpack ρ = para
    return ρ - rbfunc(J, n, para)
end

# 
function θ_fun(J::Float64, n::Float64, para::ParaCalib2)
    # Impute market tightness from J and n using free entry condition
    @unpack α, λ, A, B, Bg, ρ, σ, Bg, η, k, ξ   = para
    ra = rafunc(J, n, para)
    return (A*J/((1+ra)*k))^(1/ξ)
end

function θ_fun(J::Float64, n::Float64, para::ParaCalib2; B_shock=1.0, α_shock = 1.0, η_shock = 1.0)
    # Impute market tightness from J and n using free entry condition
    # Business cycle version
    @unpack α, λ, A, B, Bg, ρ, σ, Bg, η, k, ξ   = para
    ra = rafunc(J, n, para, B_shock, α_shock, η_shock)
    return (A*J/((1+ra)*k))^(1/ξ)
end

# Wrapper function for broadcasting
θ_fun(J, n, para, B_shock, α_shock, η_shock) = θ_fun(J, n, para; B_shock, α_shock, η_shock)


function zfunc(J::Float64, n::Float64, para::ParaCalib2)
    # Productivity 
    @unpack zbar, σ, α, λ, B, Bg, η = para
    L = min(B, η*n*J+Bg)
    return zbar + (σ/(1+σ))*(α/n)*(λ*B+(1-λ)*L)
end

function zfunc(J::Float64, n::Float64, para::ParaCalib2; x_shock=1.0, B_shock=1.0, α_shock=1.0, η_shock=1.0)
    # Business cycle version
    @unpack zbar, σ, α, λ, B, Bg, η = para
    # apply shocks
    B = B*B_shock
    α = shock_transform(α, α_shock)
    η = shock_transform(η, η_shock)

    L = min(B, η*n*J+Bg)
    return x_shock + (σ/(1+σ))*(α/n)*(λ*B+(1-λ)*L)
end

# Wrapper function for broadcasting
zfunc(J, n, para, x_shock, B_shock, α_shock, η_shock) = zfunc(J, n, para; x_shock, B_shock, α_shock, η_shock)


function wfunc(J::Float64, n::Float64, Je::Float64, ne::Float64, FixFlag::Int64, para::ParaCalib2)
    #Wage function
    @unpack A, k, ξ, ϕ, w0, ρ, δ, w1 = para

    β = 1/(1+ρ)

    θ = (A*J/(k*(1+rafunc(J,n,para))))^(1/ξ)
    q = A*θ^(-ξ)
    rap = rafunc(Je,ne,para)

    out = (1-ϕ)*w0 +ϕ*(zfunc(J,n,para)+β*(1+rap)*k*θ) + ϕ*(1-δ)*k*(1-β*(1+rap))/q
    #out = (1-ϕ)*w0 +ϕ*(zfunc(J,n,para)+k*θ)

    rout = FixFlag*w1+(1-FixFlag)*out

    return rout
end


function Jfunc(n::Float64, Je::Float64, ne::Float64, FixFlag::Int64,para::ParaCalib2)

    @unpack zbar, σ, w1, δ, B, Bg, α, λ, ρ, η= para

    function obj(x::Float64)
        # Fixed point of firm value to solve Bellman
        L = min(B, η*n*x+Bg)
        #zbar+(σ/(1+σ))*(ys(x,n,para))^(1+σ) - w1 +(1-δ)*Je/(1+rfunc(Je, ne, para))-x
        return zbar+(σ/(1+σ))*(α/n)*(λ*B+(1-λ)*L) - wfunc(x,n,Je,ne,FixFlag,para) +(1-δ)*Je/(1+rafunc(Je, ne, para))-x
    end

    sol = fzero(x -> obj(x),1.0)
    return sol
end

function Jfunc(n::Float64, Je::Float64, ne::Float64, FixFlag::Int64,para::ParaCalib2, r_fixed::Float64)

    @unpack zbar, σ, w1, δ, B, Bg, α, λ, ρ, η = para

    function obj(x::Float64)
        # fixed point of J
        L = min(B, η*n*x+Bg)
        #zbar+(σ/(1+σ))*(ys(x,n,para))^(1+σ) - w1 +(1-δ)*Je/(1+rfunc(Je, ne, para))-x
        
        return zbar+(σ/(1+σ))*(α/n)*(λ*B+(1-λ)*L) - wfunc(x,n,Je,ne,FixFlag,para) +(1-δ)*Je/(1+r_fixed)-x
    end

    sol = fzero(x -> obj(x), 12.0)
    return sol
end

function nfunc(n::Float64, Je::Float64, para::ParaCalib2)
    # Future employment
    @unpack δ, A, k, ξ = para

    function obj(x::Float64)
       # guess future employment to get future interest rate ra
       return (1-δ)*n+A^(1/ξ)*(Je/(k*(1+rafunc(Je, x, para))))^((1-ξ)/ξ)*(1-n)-x

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

function fp(FixFlag::Int64, para::ParaCalib2, ra_fixed::Float64)

    function obj(x::Array{Float64,1})

        obj1 = Jfunc(x[2], x[1], x[2], FixFlag, para, ra_fixed)-x[1]
        obj2 = nfunc(x[2],x[1],para)-x[2]
        return [obj1 obj2]
    end        

    # check high steady state
    sol1 = nlsolve(obj, [4.1;.5])
    # check low steady state
    sol2 =  nlsolve(obj,[4.1; 0.05])  
        
    return [sol1.zero  sol2.zero]
end

# Matching probabilities
function jf(θ, A, ξ)
    return min(A*θ^(1-ξ), 1.0)
end

#vacancy filling probability
function vf(θ, A, ξ)
    return min(A*θ^(-ξ), 1.0)
end

function θ_invert(q, A, ξ)
    θ = (q/A)^(-1/ξ)
    return θ
end


function SteadyState(para::ParaCalib2, FixFlag::Int64 = 0)
    " Return steady-state values as named tuple "
    @unpack A, k, ξ, ρ, Bg, η = para
    ss = fp(FixFlag, para)
    J, n = ss[:, 1]
    M = n*J
    u = 1 - n
    r_b = rbfunc(J, n, para)
    r_a = rafunc(J, n, para)
    z = zfunc(J, n, para)
    θ = (A*J/(k*(1+r_a)))^(1/ξ)
    w = wfunc(J, n, J, n, FixFlag, para)
    spread_bond = ρ - r_b
    spread_equity = ρ - r_a
    debt_gdp = Bg/(n*z)
    stock_gdp = J/z
    # output as named tuple 
    out = (n=n, J=J, M=M, u=u, θ=θ, w=w, z=z, r_a=r_a, r_b=r_b,
     spread_bond=spread_bond, spread_equity=spread_equity, x=debt_gdp, stock_gdp=J/z)
    return out
end


function ParaTrans(x::Float64, para::ParaCalib2, FixFlag::Int64)
    """ 
    Struct in terms of debt_GDP rather than Bg
    Map debt-to-GDP x to Bg
    """
    para_new = deepcopy(para)
    function output_loss(y)
        y = sqrt(y^2)
        #debt = (debt/gdp)*gdp
        Bg = x*y
        # modify instance
        para_new.Bg = Bg
        # calculate steady state
        steady = SteadyState(para, FixFlag)
        return 100*(steady.n*steady.z-y)/y, para_new
    end

    # Find root
    sol = fzero(y -> output_loss(y)[1], 0.9)
    res, para = output_loss(sol)
    return para
end

#Check steady state
# para = ParaCalib2()
# para.η = 0.5
# out = SteadyState(0, para)
# pprint(out)
# @unpack spread_bond, spread_equity, r_a, r_b, n, J, θ = out