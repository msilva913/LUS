using Plots, PyPlot
using Parameters, CSV, StatsBase, Statistics, Random
using NLsolve
using Roots, Optim, LeastSquaresOptim
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

    s::Float64 = 0.0305                 # Worker separation rate
    δ::Float64 = 0.00253                # Firm exit rate
    τ::Float64 = 1-(1-δ)*(1-s)          # Aggregate separation rate
    δ_k::Float64 = 0.008742             # Capital depreciation rate
    Z::Float64 = 1.0                    # Technology level
    b::Float64 = 1.86                   # Unemployment insurance
    ϕ::Float64 = 0.687                  # Bargaining power
    η::Float64 = 0.2                    # Elasticity of transformation cost
    ρ::Float64 = 0.00488                # Rate of time preference
    α::Float64 = 0.3                    # Capital share
    σ::Float64 = 0.05                   # Probability of preference shock
    B::Float64 = 16.656                 # Level parameter
    σ_1 = 1.5                          # elasticity of utility of output good
    σ_2 = 0.5                           # elasticity of utility of early consumption good
    λ = 0.8                             # Probability of access to perfect credit
    γ::Float64 = 0.25
    ω::Float64 = 42.322
    A::Float64 = 0.5631
    ξ::Float64 = 0.5
    Bg::Float64 = 4.529
    η_a::Float64 = 0.5
    η_k::Float64 = 0.5


    # autocorrelation coefficients
    ρx::Float64 = 0.979
    ρB::Float64 = 0.979
    ρσ::Float64 = 0.979
    ρηa::Float64 = 0.979

    # standard deviations
    σx::Float64 = 0.007
    σB::Float64 = 0.07
    σσ::Float64 = 0.07
    σηa::Float64 = 0.07

end

function solve_steady_state(x, para)
    @unpack B, σ_1, σ_2, λ, σ, s, τ, ρ, η, η_a, η_k, δ, δ_k, ω, γ, A, ξ, Bg, Z, α, ϕ, b = para
    θ, j, p = sqrt.(x.^2)
    τ = 1-(1-δ)*(1-s)

    # implied from θ
    β = 1/(1+ρ)
    f = θ -> jf(θ, A, ξ)
    q = θ -> vf(θ, A, ξ)
    n = (1-δ)*f(θ)/(τ+(1-δ)*f(θ))
    u = 1-n
    V = θ*u
    NE = δ*(V+n)

    # Use Eulers to get asset prices 

    p_b = (1+σ*(1-λ)*j)/(1+ρ)
    r_a = (1+ρ)/(1+η_a*σ*(1-λ)*j) - 1
    r_k = (1+ρ)/( 1+η_k*σ*(1-λ)*j) + δ_k - 1
    r_k = max(r_k, 1e-8)
    
    # Prices and labor productivity
    k = (α*Z/r_k)^(1/(1-α))
    K = n*k

    # Output in each sector
    Y1 = n*Z*k^α
    Y2 = n*p^(1/η)
   

    # net revenue per worker
    z = (1-α)*Z*k^(α) + η/(1+η)*p^((1+η)/η)

    #Y = Y1 + n*η/(1+η)*p^((1+η)/η)
    #z = y - r_k*k →  n*z = Y - r_k*K
    Y = n*z +r_k*K
   
    # Flow entry cost 
    κ = ω*(r_a+δ)/(1+r_a)

    # Firm value
    J = (1+r_a)/(1-δ)*(γ+κ)/(q(θ)) + ω

    # Predetermined vacancies
    Vpret = (1-δ)*((1-q(θ))*V+s*n)  

    # Stock market cap
    M = n*J + ω*Vpret
 
    # Aggregate investment, and consumption
    I = δ_k*K
    TI = I + γ*V+ω*NE
    # Y1 = C1 + TI + intermediate output
    int_out = n*p^((1+η)/η)/(1+η)
    C1 = Y1 - TI - int_out
    C1 = max(C1, 1e-10)
    
    #cons_ratio = 1 + p*σ*(B/(p*(1+j)))^(1/σ_c)
    #C1 = C/cons_ratio
    c2 = (B/(p*(1+j)))^(1/σ_2)*C1^(σ_1/σ_2)
    #C2 = (C -C1)/p
    #C2 = σ*(B/p)^(1/σ_2)*C1^(σ_1/σ_2)*((1-λ)/(1+j)^(1/σ_2)+λ)
    C2 = σ*c2
    
    # Total consumption
    C = C1 + p*C2
    # Use job creation condition to express flow profit z-w-κ
    #surplus = ((γ+κ)/q(θ))*(r_a+τ)/(1-δ)
    surplus = (J-ω)*(r_a+τ)/(1+r_a)
    # surplus = z-w-κ
    w = z - κ - surplus

    d = n*(z-w) - γ*V - ω*NE
    ν = d/r_a
  
    w_new = (1-ϕ)*b + ϕ*(z-κ+β*(1+r_a)/(1-δ)*(γ+κ)*θ) + ϕ*(1-s)*(γ+κ)/q(θ)*(1-β*(1+r_a))

    # Effective liquidity in early-consumption market
    Liq = Bg + η_a*M + η_k*(r_k+1-δ_k)*K

    # Update for c2 based on liquidity
    c2_new = min(Liq/p,  (B/p)^(1/σ_2)*C1^(σ_1/σ_2))
    j_new = (B/p)*C1^(σ_1)/c2_new^(σ_2) - 1.0

    out = zeros(3)
    out[1] = 100*(w_new-w)/w #ensures θ chosen correctly
    out[2] = 100*(Y2-C2)/C2 # market clearing of goods market 2 to pin down p
    out[3] = 100*(c2_new-c2)/c2 # ensures accuracy of j
    #out[3] = 100*(d-d_new)/d

    bond_spread = (σ*(1-λ)*j)*(1+ρ)/(1+σ*(1-λ)*j)*100*12
    equity_premium = (1+ρ)*σ*(1-λ)*j*(1-η_a)/((1+σ*(1-λ)*j)*(1+η_a*σ*(1-λ)*j))*100*12

    res = (θ=θ, J=J, q=q(θ), u=u, V=V, n=n, z=z, Y=Y, C=C, K=K, C1=C1, C2=C2, c2=c2, j=j_new, k=k, I=I, κ=κ, w=w,
     ν=ν, d=d, p_b=p_b, r_a=r_a, r_k=r_k, M=M, p=p, Liq=Liq, bond_spread=bond_spread, equity_premium=equity_premium,
     Y1=Y1, Y2=Y2, int_out=int_out, Vpret=Vpret, NE=NE)
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

    if res.j > 1e-8

        sol = LeastSquaresOptim.optimize(x -> solve_steady_state(x, para)[1], [x[1], res.j, x[2]], Dogleg())
        if sol.converged == false
            sol = LeastSquaresOptim.optimize(x -> solve_steady_state(x, para)[1], sol.minimizer, LevenbergMarquardt())
        end
        @assert sol.converged == true
        x = sol.minimizer
        res = solve_steady_state(x, para)[2]
    end
    
    return res
end

function ParaTrans(x::Float64, para::ParaCalib)
    """ 
    Struct in terms of debt_GDP rather than Bg
    Map debt-to-GDP x to Bg
    """
    para_new = deepcopy(para)

    function output_loss(Y)
        Y = sqrt(Y^2)
        #debt = (debt/gdp)*gdp
        Bg = x*Y
        # modify instance
        para_new.Bg = Bg
        # calculate steady state
        steady = steady_state(para)
        return 100*(steady.Y-Y)/Y, para_new
    end

    # Find root
    sol = fzero(Y -> output_loss(Y)[1], 5.0)
    res, para = output_loss(sol)
    return para
end

nanmean(x) = mean(filter(!isnan,x))

function calibrate_unc(Z=1.0, f=0.396, s=0.0305, rep=0.71, hiring_cost_share=0.129, annual_dep_rate=0.1, annual_dest_rate=0.03,
    σ=0.0, stock_cap_GDP=0.95, η=0.2, α=0.3)

    #Targets 
    # γ: hiring cost share 
    # ϕ: bargaining power
    # ω: market tightness (job finding and vacancy filling rates)
    # ρ: stock market cap to GDP ratio

    δ = 1-(1-annual_dest_rate)^(1/12)
    δ_k = 1-(1-annual_dep_rate)^(1/12)

    # Composite parameters    
    # Correct job finding and vacancy filling probablities
    f = f/(1-δ)
    q = (1.0 -(1.0-1/3)^4)/(1-δ)

    θ = f/q

    # Separations taken from Beveridge curve
    τ = 1- (1-δ)*(1-s)
    u = τ/(τ+f)
    n = 1 - u

    # Rates of return
    function loss(x)
        γ, ϕ, ω, ρ = @.sqrt(x^2)
        r_k = ρ + δ_k

        k = (α*Z/r_k)^(1/(1-α))
        z = (1-α)*Z*k^(α)
        b = 0.71*z
        Y = n*Z*k^(α)
   
        κ = ω*ρ/(1+ρ)
        w = (1-ϕ)*b + ϕ*(z-κ+*(γ+κ)*θ) 
        #profit_rate = n*(z-w)/Y

        J = (1+ρ)*(γ+κ)/q + ω
        M = n*J

        out = zeros(4)
        out[1] = 100*((γ+κ)/q - 1/(ρ+s)*(z-w-κ))/((γ+κ)/q)
        out[2] = 100*(n*w/Y-0.64)/0.64
        #out[3] = 100*(profit_rate - 0.06)/0.06
        out[3] = 100*(γ/q - hiring_cost_share*w)/(γ/q)
        out[4] = 100*(M/(12*Y) - stock_cap_GDP)/stock_cap_GDP
        return out, κ, z, b, Y, w, J, M
    end

    x = LeastSquaresOptim.optimize(x -> loss(x)[1], [3.0, 0.24, 31.0, 0.04/12], LevenbergMarquardt())
    sol = x.minimizer
    γ, ϕ, ω, ρ = sol
    out, κ, z, b, Y, w, J, M = loss(sol)

    # Firm value
    para = ParaCalib(δ=δ, δ_k=δ_k, s=s, η=η, γ=γ, ω=ω, b=b, ϕ=ϕ, ρ=ρ)
    return para
end



function steady_state_unc(para::ParaCalib)
    @unpack δ, δ_k, s, η, γ, ω, b, ϕ, ρ, ξ, A, α, Z = para

    β = 1/(1+ρ)
    τ = 1 - (1-δ)*(1-s)

    function loss(θ)
        f = jf(θ, A, ξ)
        q = vf(θ, A, ξ)
        n = (1-δ)*f/(τ+(1-δ)*f)
        u = 1-n
        v = θ*u
        NE = δ*(v+n)

        # Use Eulers to get asset prices 

        p_b = 1/(1+ρ)
        r_k = ρ+δ_k
        
        # Prices and labor productivity
        k = (α*Z/r_k)^(1/(1-α))
        K = n*k

        # Output in each sector
        Y = n*Z*k^α

        # net revenue per worker
        z = (1-α)*Z*k^(α)
    
        # Flow entry cost 
        κ = ω*ρ/(1+ρ)
        # Wages
        w = (1-ϕ)*b + ϕ*(z-κ+*(γ+κ)*θ) 
    

        # job creation condition
        out = 100*((γ+κ)/q - 1/(ρ+s)*(z-w-κ))/((γ+κ)/q)
        return out, f, q, u, n, v, NE, p_b, r_k, k, K, Y, z, κ, w
    end

    θ = fzero(loss, 0.51)
    out, f, q, u, n, v, NE, p_b, r_k, k, K, Y, z, κ, w = loss(θ)
    # Firm value
    J = (1+ρ)*(γ+κ)/(q) + ω
    ν = n*J/(1+ρ)
    #d = ρ*ν
    M = n*J

    # Check targets
    # hiring_cost_share = 0.129
    # stock_cap_GDP = 0.95
    # out = zeros(4)
    # out[1]  = 100*(b-0.71*z)/b
    # out[2] = 100*(n*w/Y-0.64)/0.64
    # out[3] = 100*(γ/q - hiring_cost_share*w)/(γ/q)
    # out[4] = 100*(M/(12*Y) - stock_cap_GDP)/stock_cap_GDP
    return (J=J, ν=ν, M=M, f=f, q=q, u=u, NE=NE, r_k=r_k, k=k, K=K, Y=Y, z=z, κ=κ, w=w)
end

#  para = calibrate_unc()
#  steady = steady_state_unc(para)

#  steady.Y*12*0.4

















   