using Plots, PyPlot
using Parameters, CSV, StatsBase, Statistics, Random
using NLsolve
using LinearAlgebra, Roots, Optim, LeastSquaresOptim
using PrettyPrinting, Debugger

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

@with_kw mutable struct ParaCalib

    s::Float64 = 0.02
    δ::Float64 = 0.03045
    τ::Float64 = 1-(1-δ)*(1-s)
    δ_k::Float64 = 0.03
    Z::Float64 = 1.0
    b::Float64 = 0.6113
    w::Float64 = 0.97
    ϕ::Float64 = 0.3
    ρ::Float64 = 0.00341
    α::Float64 = 0.3
    σ::Float64 = 0.01
    B::Float64 = 16.656
    σ_c::Float64 = 1.5
    γ::Float64 = 4.0
    ω::Float64 = 3.0
    A::Float64 = 0.5631
    ξ::Float64 = 0.5
    Bg::Float64 = 4.529
    η_a::Float64 = 0.5
    η_k::Float64 = 0.5


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



function solve_steady_state(x, para)
    @unpack B, σ_c, σ, s, τ, ρ, η_a, η_k, δ, δ_k, ω, γ, A, ξ, Bg, Z, α, ϕ, b, Bg = para
    θ, j, p = sqrt.(x.^2)

    # implied from θ
    β = 1/(1+ρ)
    f = θ -> jf(θ, A, ξ)
    q = θ -> vf(θ, A, ξ)
    n = (1-δ)*f(θ)/(τ+(1-δ)*f(θ))
    u = 1-n
    v = θ*u
    N_E = δ*(v+n)

    # Use Eulers to get asset prices 

    p_b = (1+σ*j)/(1+ρ)
    r_a = (1+ρ)/(1+η_a*σ*j) - 1
    r_k = (1+ρ)/( 1+η_k*σ*j) + δ_k - 1
    r_k = max(r_k, 1e-8)
    
    # Prices and labor productivity
    k = (α*Z/r_k)^(1/(1-α))*(1+p^(1/(1-α)))
    K = n*k

    # Output in each sector
    Y1 = n*Z*(α*Z/r_k)^(α/(1-α))
    Y2 = n*Z*(α*p*Z/r_k)^(α/(1-α))
    Y = Y1 + p*Y2

    # net revenue per worker
    z =  Y/n - r_k*k
   
    # Flow entry cost 
    κ = ω*r_a/(1+r_a)

    # Firm value
    J = (1+r_a)*(γ+κ)/(q(θ)) + ω
    ν = n*J/(1+r_a)
    d = r_a*ν

    # Aggregate investment, and consumption
    I = δ_k*K
    TI = I + γ*v+κ*N_E
    C1 = Y1 - TI

    #cons_ratio = 1 + p*σ*(B/(p*(1+j)))^(1/σ_c)
    #C1 = C/cons_ratio
    c2 = (B/(p*(1+j)))^(1/σ_c)*C1
    #C2 = (C -C1)/p
    C2 = σ*c2
    
    # Total consumption
    C = C1 + p*C2
    # Use job creation condition to express flow profit z-w-κ
    surplus = ((γ+κ)/q(θ))*(r_a+s)
    # surplus = z-w-κ
    w = z - κ - surplus

    # Update of profits
    d_new = n*(z-w) - γ*v - κ*N_E
    out = zeros(3)
    w_new = (1-ϕ)*b + ϕ*(z-κ+β*(1+r_a)*(γ+κ)*θ) + (1-s)*(γ+κ)/q(θ)*ϕ*(1-β*(1+r_a))

    Liq = Bg + η_a*n*J + η_k*(r_k+1-δ_k)*K
    c2_new = min(Liq/p,  (B/p)^(1/σ_c)*C1)
    j_new = B*(c2_new/C1)^(-σ_c)/p - 1.0

    # check == 1 -> not liquidity constrained

    out[1] = 100*(w_new-w)/w
    out[2] = 100*(Y2-C2)/C2
    out[3] = 100*(c2_new-c2)/c2

    bond_spread = (σ*j)*100*12

    res = (θ=θ, u=u, v=v, n=n, N_E=N_E, z=z, Y=Y, C=C, K=K, C1=C1, C2=C2, c2=c2, j=j_new, k=k, I=I, κ=κ,
     ν=ν, d=d, p_b=p_b, r_a=r_a, r_k=r_k, M=n*J, p=p, Liq=Liq, bond_spread=bond_spread, Y1=Y1, Y2=Y2)
    return out, res
end

function solve_steady_state_unc(x, para)
    out, res = solve_steady_state([x[1],0.0, x[2]], para)
    out = [out[1] out[2]]
    return out, res
end



function steady_state(para)

    # First, check if there is an unconstrained equilibrium

    sol = LeastSquaresOptim.optimize(x -> solve_steady_state_unc(x, para)[1], [0.5, 0.8], Dogleg())
    @assert sol.converged == true
    x = sol.minimizer
    res = solve_steady_state_unc(x, para)[2]

    if res.j > 1e-5

        sol = LeastSquaresOptim.optimize(x -> solve_steady_state(x, para)[1], [x[1], res.j, x[2]], Dogleg())
        @assert sol.converged == true
        x = sol.minimizer
        res = solve_steady_state(x, para)[2]
    end
    
    return res

end


function comparative_statics(para, symbol, vals)
    θ_vals = zero(vals)
    u_vals = zero(vals)
    K_vals = similar(vals)
    Y_vals = similar(vals)
    #j_vals = similar(vals)
    bond_spread_vals = similar(vals)
    C2_vals = similar(vals)
    C1_vals = similar(vals)
    c2_vals = similar(vals)
    z_vals = similar(vals)
    ra_vals = similar(vals)
    M_vals = similar(vals)
    p_vals = similar(vals)
    Liq_vals = similar(vals)
    Y1_vals = similar(vals)
    Y2_vals = similar(vals)
    par = deepcopy(para)

    #break_on(:error)
    #@run
    for (i, x) in enumerate(vals)
        setfield!(par, symbol, x)
        steady = steady_state(par)
        θ_vals[i] = steady.θ
        C2_vals[i] = steady.C2
        C1_vals[i] = steady.C1
        c2_vals[i] = steady.c2
        z_vals[i] = steady.z
        u_vals[i] = steady.u
        Y_vals[i] = steady.Y 
        K_vals[i] = steady.K 
        #j_vals[i] = steady.j
        bond_spread_vals[i] = steady.bond_spread
        ra_vals[i] = steady.r_a
        M_vals[i] = steady.M
        p_vals[i] = steady.p
        Liq_vals[i] = steady.Liq
        Y1_vals[i] = steady.Y1
        Y2_vals[i] = steady.Y2
    end
    # Adjust using base-year prices
    p_star = p_vals[1]
    Y_vals = @. Y1_vals + p_star*Y2_vals 
    C_vals = @. C1_vals + p_star*C2_vals

    return [vals, θ_vals, u_vals, Y_vals, K_vals, bond_spread_vals, ra_vals, M_vals, p_vals, C_vals, C1_vals, c2_vals, Liq_vals]
end

function plot_statics(out, symbol; savefig = "", levels=true)
    vals, θ_vals, u_vals, Y_vals, K_vals, bond_spread_vals, ra_vals, M_vals, p_vals, C_vals, C1_vals, c2_vals, Liq_vals = out
    if levels
        vars =  [vals, θ_vals, u_vals, Y_vals, K_vals, bond_spread_vals, ra_vals, M_vals, p_vals, C_vals, C1_vals, c2_vals]
    else
        vars = [100*log.(x/x[1]) for x in (vals, θ_vals, u_vals, Y_vals, K_vals, bond_spread_vals, ra_vals, M_vals, p_vals,
                                     C_vals, C1_vals, c2_vals)]
    end
    labels = [symbol, :θ, :u, :Y, :K, :Bond_spread, :ra, :M, :p, :C, :C1, :c2, :Liquidity]

    fig = plt.figure(figsize=(16, 8))
    for (i, key) in enumerate(vars)
        var = vars[i]
        lab = labels[i]
        ax = fig.add_subplot(3, 4, i)
        ax.plot(var, linewidth=2)
        ax.set_title(lab)
        if !levels
         ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        end
        #ax.legend(loc="upper right")
    end
    plt.tight_layout()
    display(fig)
    if !isempty(savefig)
        plt.savefig("comp_statics_"*savefig)
    end
end


para = ParaCalib(δ=0.02, Bg=0.0, s=0.035, η_a=0.5, η_k=0.2, B=10.0, γ=3.0, ω=20.0, σ_c=0.9, σ=0.05)
steady = steady_state(para)


Bg_vals = 0:0.1:2
out = comparative_statics(para, :Bg, Bg_vals)
vals, θ_vals, u_vals, Y_vals, K_vals, j_vals, ra_vals, M_vals, p_vals, C_vals, c2_vals, Liq_vals = out

B_vals = 1:0.1:20
out = comparative_statics(para, :B, B_vals)
vals, θ_vals, u_vals, Y_vals, K_vals, bond_spread_vals, ra_vals, M_vals, p_vals, C_vals, C1_vals, c2_vals, Liq_vals = out

plot_statics(out, :B)

η_k_vals = 0.0:0.05:0.4
out = comparative_statics(para, :η_k, η_k_vals)
vals, θ_vals, u_vals, Y_vals, K_vals, j_vals, ra_vals, M_vals, p_vals, C_vals, c2_vals, Liq_vals = out

σ_vals = 0.0001:0.005:0.05
out = comparative_statics(para, :σ, σ_vals)
vals, θ_vals, u_vals, Y_vals, K_vals, bond_spread_vals, ra_vals, M_vals, p_vals, C_vals, C1_vals, c2_vals, Liq_vals = out

plot_statics(out, :σ)

Z_vals = 1:0.1:1.4
out = comparative_statics(para, :Z, Z_vals)
vals, θ_vals, u_vals, Y_vals, K_vals, j_vals, ra_vals, M_vals, p_vals, C_vals, c2_vals, Liq_vals = out










   