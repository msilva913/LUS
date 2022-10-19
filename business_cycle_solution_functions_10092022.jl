using Random: seed!
using LinearAlgebra: diagm, cholesky, dot
using InteractiveUtils: versioninfo
using DelimitedFiles: writedlm

using Sobol
using BasisMatrices, QuantEcon, Roots, DataFrames, MAT, Parameters
using JSON3
using Printf
using Statistics
include("steady_state_convexity.jl")

columns(M) = (view(M, :, i) for i in 1:size(M, 2))

function shock_transform(ss_val, shock)
    return ss_val*shock/(1-ss_val+ss_val*shock)
end

cd("C:\\Users\\TJSEM\\Dropbox\\Unemployment Stocks II\\JEDC revision\\Programs\\Julia 25 July 2022")
#cd("C:\\Users\\BIZtech\\Dropbox\\Unemployment Stocks II\\JEDC revision\\Programs\\Julia 25 July 2022")

json_string = read("params_calib.json", String)
par = JSON3.read(json_string)
s, δ, δ_k, Z, b, ϕ, η, ρ, α, σ, B, σ_1, σ_2, λ, γ, ω, A, ξ, Bg, η_a, η_k = par
para = ParaCalib(s=s, δ=δ, δ_k=δ_k, Z=Z, b=b, ϕ=ϕ, η=η, ρ=ρ, σ=σ, B=B, σ_1=σ_1, σ_2=σ_2, λ=λ,
    γ=γ, ω=ω, A=A, ξ=ξ, Bg=Bg, η_a=η_a, η_k=η_k)
ss = steady_state(para)


function Params(para;ρx=0.979, σx=0.007, 
    ρB=0.979, σB=0.007, 
    ρσ=0.979, σσ=0.007, 
    ρηa=0.979, σηa=0.007)

    # modify with shocks
    par = deepcopy(para)
    par.ρx = ρx
    par.σx = σx
    par.ρB = ρB
    par.σB = σB
    par.ρσ = ρσ
    par.σσ = σσ
    par.ρηa = ρηa
    par.σηa = σηa
    
    return par
end

function grid_size(p, deg)
    Dict(1 =>20, 2 =>200, 3 => 300, 4 => 1000, 5 => 2000)[deg]
end

# returns covariance matrix for 6 shocks in the model
vcov(p) = diagm(0 => [p.σx^2, p.σB^2, p.σσ^2, p.σηa^2])


# Given an instance of Params and SteadyState, construct the grid for solving the model
function Grids(p, ss, deg=2)
    m = grid_size(p, deg)
    σ = [p.σx, p.σB, p.σσ, p.σηa]
    ρ = [p.ρx, p.ρB, p.ρσ, p.ρηa]

    Vpret_min, Vpret_max = 0.25*ss.Vpret, ss.Vpret*2
    K_min, K_max = 0.25*ss.K, 1.5*ss.K
    u_min, u_max = 0.025, 0.25

    ub = [2 * p.σx / sqrt(1 - p.ρx^2),
          2*p.σB/sqrt(1-p.ρB^2),
          2*p.σσ/sqrt(1-p.ρσ^2),
          2*p.σηa/sqrt(1-p.ρηa^2),
           Vpret_max, K_max, u_max]
    lb = -ub
    lb[5] = Vpret_min
    lb[6] = K_min
    lb[7] = u_min

    # construct SobolSeq
    s = SobolSeq(length(ub), ub, lb)
   
    seq = zeros(m, length(lb))

    for i in 1:m
        seq[i, :] = next!(s)
    end
    #exogenous states (x, B, σ, ηa)
    ξ = seq[:, 1:4]
    # endogenous states: (u, Vpret, K)
    Vpret, K, u = columns(seq[:, 5:7])
   

    # decompose shocks
    ξx = ξ[:, 1]
    ξB = ξ[:, 2]
    ξσ = ξ[:, 3]
    ξηa = ξ[:, 4]

    # Store grid
    X = [Vpret K u ξx ξB ξσ ξηa]
    # complete polynomial
    X0_G = Dict(
        1 => complete_polynomial(X, 1),
        deg => complete_polynomial(X, deg)
    )

    ϵ_nodes, ω_nodes = qnwmonomial1(vcov(p))

    ξx1 = p.ρx.*ξx .+ ϵ_nodes[:, 1]'
    ξB1 = p.ρB.*ξB .+ ϵ_nodes[:, 2]'
    ξσ1 = p.ρσ.*ξσ .+  ϵ_nodes[:, 3]'
    ξηa1 = p.ρηa.*ξηa .+ ϵ_nodes[:, 4]'

    Grids = (ξx, ξB, ξσ, ξηa, u, Vpret, K, X, X0_G, ϵ_nodes, ω_nodes, 
    ξx1, ξB1, ξσ1, ξηa1, Vpret_min=Vpret_min, Vpret_max=Vpret_max, K_min=K_min, K_max=K_max, u_min=u_min, u_max=u_max)
    return Grids
end

# Construct Model type, which has instance of Params, SteadyState, and Grids

function Model(p; deg=2)
    #p = Params()
    s = steady_state(p)
    g = Grids(p, s, deg)
    Model = (p=p, s=s, g=g, deg=deg)
    return Model
end

Base.show(io::IO, m) = println(io, "LUS model")


# Helper function with takes control variables S, F, C and shocks δ, R, η_G, η_a, η_L, η_R and applied equilibrium conditions to
# produce   π, δ', Y, L, Y_n, R'

function accuracy_check(m)
    ss = m.s
    @unpack B, σ_1, σ_2, λ, σ, s, τ, ρ, η, η_a, η_k, δ, δ_k, ω, γ, A, ξ, Bg, Z, α, ϕ, b = m.p
    @unpack θ, J, C1, w, Vpret, K, u, κ, r_a = ss

    out = Base.step(m, θ, J, C1, κ, 1+r_a, Vpret, K, u, 0.0, 0.0, 0.0, 0.0)
    Vpret_p, K_p, u_p, M, r_k, p, w, z, q, V, NE, c2, C, Y, I, j, d, ν, B_shock, σ_shock, ηa_shock = out
    res = zeros(15)
    res[1] = Vpret_p - Vpret
    res[2] = K_p - K
    res[3] = u_p - u
    res[4] = r_k - ss.r_k 
    res[5] = p - ss.p 
    res[6] = z - ss.z 
    res[7] = q - ss.q 
    res[8] = V - ss.V 
    res[9] = NE - ss.NE
    res[10] = C - ss.C 
    res[11] = Y - ss.Y 
    res[12] = I - ss.I 
    res[13] = j - ss.j 
    res[14] = d - ss.d 
    res[15] = ν - ss.ν
    print("Max error is $(maximum(res))")
    return res
end

function Base.step(m, θ, J, C1, κ, ra_exp, Vpret, K, u, ξx, ξB, ξσ, ξηa)
    p, g = m.p, m.g
    # Helper function which updates quantities from J, u_lag
    @unpack B, σ_1, σ_2, λ, σ, s, τ, ρ, η, η_a, η_k, δ, δ_k, ω, γ, A, ξ, Bg, Z, α, ϕ, b = m.p

    β = 1/(1+ρ)

    # Express shocks in levels
    x_shock = exp.(ξx)
    B_shock= exp.(ξB)
    σ_shock = exp.(ξσ)
    ηa_shock = exp.(ξηa)

    # Transform shocks
    B_shock = B*B_shock
    σ_shock = shock_transform.(σ, σ_shock)
    ηa_shock = shock_transform.(η_a, ηa_shock)

    n = @. 1 - u
    r_k = @. α*x_shock*(n/K)^(1-α)

    # Stock market capitalization
    M = @. n*J + ω*Vpret

    # Effective liquidity
    L = @. Bg + ηa_shock*M + η_k*(r_k+1-δ_k)*K

    # Calculate constrained price 
    p_c = @. ((σ_shock/n)*L)^(η/(η+1))
    p_un = @. (σ_shock/n*B_shock^(1/σ_2)*C1^(σ_1/σ_2))^(η*σ_2/(η+σ_2))

    # Check if constraint binds
    cons = @. (L/p_c) < (B_shock/p_c)^(1/σ_2)*C1^(σ_1/σ_2)
    
    # Correct price depending on whether constraint binds
    p = @. cons*p_c + (1-cons)*p_un
    # Marginal revenue product of labor
    z = @. (1-α)*x_shock*(K/n)^(α) + η/(1+η)*p^((1+η)/η)

    # Consumption of second good
    c2 = @. n*p^(1/η)/σ_shock

    # Aggregate consumption and output
    C = @. C1 + n*p^((η+1)/η)
    Y = @. n*z+r_k*K
    θ = @. min(θ, (Y+ω*Vpret-C)/((γ+ω)*u) )
    θ = @. max(θ, Vpret/u)
    # Vacancy filling rate and tightness
    f = jf.(θ, A, ξ)
    q = vf.(θ, A, ξ)

    # Vacancies and entrants
    V = @. θ*(1-n)
    NE = @. V - Vpret

    # # Zero lower bound on entrants
    # NE = @. max(NE, 0.0)
    # V = @. Vpret + NE
    # θ = @. V/u
    # f = jf.(θ, A, ξ)
    # q = vf.(θ, A, ξ)

    # Aggregate consumption, output, and investment
    
     I = @.  Y - C - γ*V - ω*NE
     I = @. max(I, 0.0)
    

    # Marginal surplus of liquidity
    j = @. (B_shock/p)*(C1^σ_1)/(c2^(σ_2)) - 1

    # Wage 
    w = @. (1-ϕ)*b + ϕ*(z-κ+β*ra_exp/(1-δ)*(γ+κ)*θ) + ϕ*(1-s)*(γ+κ)/q*(1-β*ra_exp)

    # Dividends and firm values
    d = @. n*(z-w) - γ*V - ω*NE
    d = max.(d, -n.*J./2)
 
    ν = @. M - d

    # Update state variables
    n_p = @. (1-δ)*((1-s)*n+f*(1-n))
    u_p = @. 1 - n_p
    Vpret_p = @. (1-δ)*((1-q)*V + s*n)
    K_p = @. (1-δ_k)*K + I

    # Check bounds
    K_p = min.(K_p, g.K_max)
    K_p = max.(K_p, g.K_min)
    Vpret_p = min.(Vpret_p, g.Vpret_max)
    Vpret_p = max.(Vpret_p, g.Vpret_min)
    u_p = min.(u_p, g.u_max)
    u_p = max.(u_p, g.u_min)

    return Vpret_p, K_p, u_p, M, r_k, p, w, z, q, V, NE, c2, C, Y, I, j, d, ν, B_shock, σ_shock, ηa_shock
end


# Solution routine

function initial_coefs(m, deg)
    npol = size(m.g.X0_G[deg], 2)
    coefs = zeros(npol, 5)
    # initialize at steady-state values (non-constant terms are zero)
    coefs[1, :] = [m.s.θ, m.s.J, m.s.C1, m.s.κ, m.s.r_a + 1.0]
    return coefs
end


function solve(m;damp=0.3, deg=2, tol=1e-7, verbose::Bool=true, coefs_init::Union{Matrix{Float64}, Nothing}=nothing)
    # simplify notation
    @unpack B, σ_1, σ_2, λ, σ, s, τ, ρ, η, η_a, η_k, δ, δ_k, ω, γ, A, ξ, Bg, Z, α, ϕ, b = m.p

    β = 1.0/(1.0+ρ)

    p, g, deg = m.p, m.g, m.deg
    n, n_nodes = size(m.g.ξx, 1), length(m.g.ω_nodes)
    # number of policies to solve for
    npolicies = 5 # θ, J, C1, κ, w
    nstates = size(m.g.X)[2]
    ## allocate memory
    # euler equations
    e = zeros(n, npolicies)

    # previous policies
    θ_old = ones(n)
    J_old = ones(n)
    C1_old = ones(n)
    κ_old = ones(n)
    ra_exp_old = ones(n)

    # updated policies
    θ_new = similar(J_old)
    J_new = similar(J_old)
    C1_new = similar(J_old)
    κ_new = similar(J_old)
    ra_exp_new = similar(J_old)


    # future policies
    θ1 = zeros(n, n_nodes)
    J1 = zeros(n, n_nodes)
    C11 = zeros(n, n_nodes)
    κ1 = zeros(n, n_nodes)
    ra_exp1 = zeros(n, n_nodes)

    ra_1 = zeros(n, n_nodes)
    hire_cost = zeros(n)

    local coefs, start_time
    # set up matrices for this degree
    X0_G = g.X0_G[deg]

    # future basis matrix for policies
    X1 = Array{Float64}(undef, n, n_complete(nstates, deg))

    if coefs_init === nothing
        coefs = initial_coefs(m, deg)
    else
        coefs = coefs_init
    end

    error = 1.0
    it = 0

    start_time = time()
    # solve at this degree of complete polynomial
    while error > tol
        it += 1;
        # current choices (at t)
        # --------------------------------
        θ = sqrt.((X0_G*coefs[:, 1]).^2)
        J = sqrt.((X0_G*coefs[:, 2]).^2)
        C1 = sqrt.((X0_G*coefs[:, 3]).^2)
        κ = X0_G*coefs[:, 4]
        ra_exp = sqrt.((X0_G*coefs[:, 5]).^2)
    
        out = step(m, θ, J, C1, κ, ra_exp, g.Vpret, g.K, g.u, g.ξx, g.ξB, g.ξσ, g.ξηa)
        Vpret1, K1, u1, M0, r_k0, p0, w0, z0, q0, V0, NE0, c20, C0, Y0, I0, j0, d0, ν0, B0_shock, σ0_shock, ηa0_shock = out
        
      

        for node in 1:n_nodes
            # Form complete polynomial of degree "Degree" at t+1 on future state 
            grid1 = hcat(Vpret1, K1, u1, g.ξx1[:, node], g.ξB1[:, node], g.ξσ1[:, node], g.ξηa1[:, node])
            complete_polynomial!(X1, grid1, deg)
            
            θ1[:, node] = sqrt.((X1*coefs[:,1]).^2)
            J1[:, node] = sqrt.((X1*coefs[:,2]).^2)
            C11[:, node] = sqrt.((X1*coefs[:, 3]).^2)
            κ1[:, node] = X1*coefs[:, 4]
            ra_exp1[:, node] = sqrt.((X1*coefs[:, 5]).^2)

        end

        # Next-period quantities
        out = step(m, θ1, J1, C11, κ1, ra_exp1, Vpret1, K1, u1, g.ξx1, g.ξB1, g.ξσ1, g.ξηa1)

        Vpret2, K2, u2, M1, r_k1, p1, w1, z1, q1, V1, NE1, _, _, Y1, I1, j1, d1, ν1, B1_shock, σ1_shock, ηa1_shock = out
      
        # Evaluate conditional expectations in the Euler equations
        #---------------------------------------------------------
        #(n x n_nodes)*(n_nodes x 1)
        # Calculate J using Euler equation and update theta

        #1) Job creation curve and J
        ra_1 .= @. (ν1+d1)/ν0 - 1.0
        #ra_1 .= max.(ra_1, -δ)

        hire_cost .= @.(((1-δ)/(1+ra_1))*(z1-w1-κ1 + (1-s)*(γ+κ1)/q1))*g.ω_nodes
        J_new .= @. ω + z0 - w0 - κ + (1-s)*hire_cost


        #2) Euler equation for capital and C1
        C1_new .= (@.(β*C11^(-σ_1)*(1-δ_k+r_k1))*g.ω_nodes).^(-1/σ_1)


        #3) Euler equation for equity
        lhs = @.(β*C11^(-σ_1)*(ν1+d1)*(1+ηa1_shock*σ1_shock*j1))*g.ω_nodes
        ν = lhs./(C1.^(-σ_1))
        J = @. (ν0+d0-ω*g.Vpret)/n
        hire_cost = @. (J-ω - (z0-w0-κ))/(1-s)
        q = @. 1/(hire_cost)*(γ+κ)
        q .= min.(q, 0.99)
        q .= max.(q, 0.01)
        θ_new .= jf.(θ, A, ξ)./q

        #4) Euler equation for κ
        κ_new .= @.(ω*(ra_1+δ)/(1+ra_1))*g.ω_nodes

        #5) Euler equation for ra_exp
        ra_exp_new .= @.(1+ra_1)*g.ω_nodes

        # Update wage using wage equation
        # Update J using Firm Bellman
        e[:, 1] .= θ_new
        e[:, 2] .= J_new
        e[:, 3] .= C1_new
        e[:, 4] .= κ_new
        e[:, 5] .= ra_exp_new
      
        # Variables of the current iteration
        #-----------------------------------
        # Compute and update the coefficients of the decision functions
        # -------------------------------------------------------------
        coefs_hat = X0_G\e   # Compute the new coefficients of the decision
                                # functions using a backslash operator

        # Update the coefficients using damping
        coefs = damp*coefs_hat + (1-damp)*coefs

        # Evaluate the percentage (unit-free) difference between the values
        # on the grid from the previous and current iterations
        # -----------------------------------------------------------------
        # The convergence criterion is adjusted to the damping parameters
        error = maximum(abs, 1.0 .- θ_new./θ_old) +
                maximum(abs, 1.0 .- J_new./J_old) +
                maximum(abs, 1.0 .- C1_new./C1_old) +
                maximum(abs, 1.0 .- κ_new./κ_old) +
                maximum(abs, 1.0 .- ra_exp_new./ra_exp_old)

        if (it % 20 == 0) && verbose
            @printf "On iteration %d err is %6.7e\n" it error
        end

        # Store the obtained values for E_t on the grid to
        # be used on the subsequent iteration in Section 10.2.6
        #-----------------------------------------------------------------------
        copy!(θ_old, θ_new)
        copy!(J_old, J_new)
        copy!(C1_old, C1_new)
        copy!(κ_old, κ_new)
        copy!(ra_exp_old, ra_exp_new)

    end
    return coefs, time() - start_time
end


function Simulation(m, coefs::Matrix; capT=50_000, burn_in=500)
    # 11) Simulating a time sries Solution
    seed!(1234)
    s = m.s
    T = capT + burn_in
    #----------------------------------------
    rands = randn(T, 4)
     # Initialize the values of 6 exogenous shocks
    #--------------------------------------------
    ξx = zeros(T)
    ξB = zeros(T)
    ξσ = zeros(T)
    ξηa = zeros(T)

    # Generate the series for shocks
    #-------------------------------
    @inbounds for t in 1:T-1
        ξx[t+1] = m.p.ρx*ξx[t] + m.p.σx*rands[t, 1]
        ξB[t+1] = m.p.ρB*ξB[t] + m.p.σB*rands[t, 2]
        ξσ[t+1] = m.p.ρα*ξσ[t] + m.p.σα*rands[t, 3]
        ξηa[t+1] = m.p.ρηa*ξη[t] + m.p.σηa*rands[t, 4]
    end
    #u1, Vpret1, K1, r_k0, p0, z0, θ0, q0, V0, NE0, C0, Y0, I0, j0, d0, ν0

    # Initialize states 
    u = ones(T).*s.u
    Vpret = ones(T).*s.Vpret
    K = ones(T).*s.k
    
    vars = Array{Float64}(undef, T, 20)
    J, C1, M, κ, w, p, θ, q, V, NE, c2, C, Y, I, j, d, ν, B_shock, σ_shock, ηa_shock =  columns(vars)


    pol_bases = Array{Float64}(undef, 1, size(coefs, 1))
    # first period accounts for lagged unemployment; proper time series starts at 2
    @inbounds for t in 1:(T)
        # construct the matrix of explanatory variables "pol_bases" on the series of state variables
        complete_polynomial!(
            pol_bases,
            # lagged unemployment and productivity
            hcat(Vpret[t], K[t], u[t], ξx[t], ξB[t], ξσ[t], ξηa[t]),
            m.deg
        )
        # Extract policy function
        policies = pol_bases*coefs
        θ[t],  J[t], C1[t], κ[t], ra_exp[t] = policies
        out = step(m, J[t], C1[t], κ[t], w[t], Vpret[t], K[t], u[t], ξx[t], ξB[t], ξσ[t], ξηa[t])
        Vpret[t+1], K[t+1], u[t+1], M[t], r_k[t], p[t], z[t], q[t], V[t], NE[t], C[t], Y[t], I[t], j[t], d[t], ν[t], B_shock[t], σ_shock[t], ηa_shock[t] = out
    end

    # remove last element of state variables
    pop!(Vpret)
    pop!(u)
    pop!(K)

    f = @.θ*q
    y = @. (1-u)*z
    out = [M J u C1 κ w p θ q V NE c2 C Y I j d  ξx ξB ξσ ξηa][(burn_in+1):(end), :]
    J, C1, κ, w, p, θ, q, V, NE, C, Y, I, j, d, ξx, ξB, ξσ, ξηa = columns(out)
    Simulation = (M=M, J=J, u=u, C1=C1, κ=κ, w=w, p=p, θ=θ, q=q, V=V, NE=NE, c2=c2, C=C, Y=Y, I=I, j=j, d=d,
                    ξx=ξx, ξB=ξB, ξσ=ξσ, ξηa=ξηa)
    return Simulation
end

function Residuals(m, coefs::Matrix, sim)
    p, g = m.p, m.g
    @unpack zbar, σ, w1, δ, B, Bg, α, λ, ρ, η, w0, A, ξ, ϕ, k = p
    β = 1/(1+ρ)
    capT = length(sim.w)
    resids = zeros(4, capT-1)

    # Integration method for evaluating accuracy
    # ------------------------------------------
    # Monomial integration rule with 2N^2+1 nodes
    ϵ_nodes, ω_nodes = qnwmonomial2(vcov(m.p))
    n_nodes = length(ω_nodes)

    # Allocate for arrays needed in the loop
    basis_mat = Array{Float64}(undef, n_nodes, 6)
    X1 = Array{Float64}(undef, n_nodes, size(coefs, 1))

    ξx1 = Array{Float64}(undef, n_nodes)
    ξB1 = Array{Float64}(undef, n_nodes)
    ξα1 = Array{Float64}(undef, n_nodes)
    ξη1 = Array{Float64}(undef, n_nodes)

    for t in 1:(capT-1)                 # For each given point,
        # Take the corresponding value for shocks at t
        #---------------------------------------------
        ξx = sim.ξx[t]  # ηR(t)
        ξB = sim.ξB[t]
        ξα = sim.ξα[t]
        ξη = sim.ξη[t]

        # Extract time t values for all other variables (and t+1 for R, δ)
        #------------------------------------------------------------------
        u  = sim.u[t]
        θ = sim.θ[t]
        J = sim.J[t]
        w = sim.w[t]
        z = sim.z[t]
        q = sim.q[t]
        # Fill basis matrix with R1, δ1 and shocks
        #-----------------------------------------
        # Note that we do not premultiply by standard deviations as ϵ_nodes
        # already include them. All these variables are vectors of length n_nodes
        copy!(ξx1, ξx*p.ρx .+ ϵ_nodes[:, 1])
        copy!(ξB1, ξB*p.ρB .+ ϵ_nodes[:, 2])
        copy!(ξα1, ξα*p.ρα .+ ϵ_nodes[:, 3])
        copy!(ξη1, ξη*p.ρη .+ ϵ_nodes[:, 4])

        basis_mat[:, 1] .= u
        basis_mat[:, 2] .= log.(u)

        basis_mat[:, 3] .= ξx1
        basis_mat[:, 4] .= ξB1
        basis_mat[:, 5] .= ξα1
        basis_mat[:, 6] .= ξη1

        # Future choices at t+1
        #----------------------
        # Form a complete polynomial of degree "Degree" (at t+1) on future state
        # variables; n_nodes-by-npol
        complete_polynomial!(X1, basis_mat, m.deg)

        # Compute next-period policies
        J1 = (X1*coefs[:,1]).^2
        θ1 = (X1*coefs[:, 1]).^2
        out = step(m, J1, θ1, u, ξx1, ξB1, ξα1, ξη1)
        u1, q1, __, ra1, rb1, v1, z1 = out
         
        # Compute residuals 
        # Revenue function
        resids[1, t] = 1 - zfunc(J, 1-u, p, exp.(ξx), exp.(ξB), exp.(ξα), exp.(ξη))/z
        #z = zfunc(J, 1-u, exp(ηx), p)
        # Euler equation for firms
        resids[2, t] = 1 - dot(ω_nodes, @.(z -w +(1-δ)*J1/(1+ra1)))/J

        # Free entry condition
        resids[3, t] = 1.0 - θ_fun(J,1-u,p, exp.(ξB), exp.(ξα), exp.(ξη))/θ
        # Wage equation
        resids[4, t] = 1.0 - dot(ω_nodes, @.((1-ϕ)*w0 + ϕ*(z+β*(1+ra1)*k*θ) + ϕ*(1-δ)*k/q*(1-β*(1+ra1))))/w
    end
    # discard the first burn observations
    return Residual(resids)
end

function impulse_response(m, coefs; shock_type="x", irf_length=60, scal=1.0)
    p = m.p

    x_series, B_series, α_series, η_series = columns(ones(irf_length, 3))
    x_bas, B_bas, α_bas, η_bas = columns(ones(irf_length, 3))

    shock = zeros(irf_length)

    shock[1] = p.σx*scal
    # Simulate shocks
    if shock_type == "x"
        ρ = p.ρx
    elseif shock_type == "B"
        ρ = p.ρB
    elseif shock_type == "α"
        ρ = p.ρα
    elseif shock_type == "η"
        ρ = p.ρη
    end

    # generate shock
    for t in 1:(irf_length-1)
        shock[t+1] = ρ*shock[t]
    end

    shock = exp.(shock)

    if shock_type == "x"
        x_series = shock
    elseif shock_type == "B"
        B_series = shock
    elseif shock_type == "α"
        α_series = shock
    end

    pol_bases = Array{Float64}(undef, 1, size(coefs, 1))

    function impulse(x_series, B_series, α_series, η_series)
        #
        u = zeros(irf_length)
        u[1] = m.s.u
        A = Array{Float64}(undef, irf_length, 13)
        θ, w, J, q, f, ra, rb, v, z, M, equity_spread, bond_spread, Y = columns(A)
        for t in 2:irf_length
            complete_polynomial!(
                pol_bases,
                hcat(u[t-1], log(u[t-1]), log(x_series[t]), log(B_series[t]), log(α_series[t]), log(η_series[t])),
                m.deg
            )
            # Extract policy function
            policies = pol_bases*coefs
            J[t], θ[t], w[t] = policies[1]^2, policies[2]^2, policies[3]
            out = step(m, J[t], θ[t], u[t-1], log(x_series[t]), log(B_series[t]), log(α_series[t]), log(η_series[t]))
            u[t], q[t], __, ra[t], rb[t], v[t], z[t] = out
            equity_spread[t] = p.ρ - ra[t]
            bond_spread[t] = p.ρ - rb[t]
            M[t] = (1-u[t])*J[t]
            Y[t] = (1-u[t])*z[t]
            f[t] = θ[t]*q[t]

        end
        out = [u θ q f ra rb v z M w bond_spread equity_spread Y][2:end, :]
        return out
    end


    out_imp = impulse(x_series, B_series, α_series, η_series)
    out_bas = impulse(x_bas, B_bas, α_bas, η_bas)

    # Percentage deviations (100*log deviations)
    irf_res = @. 100*log(out_imp/out_bas)
    u, θ, q, f, ra, rb, v, z, M, w, bond_spread, equity_spread, Y = columns(irf_res)
    irf = (u=u, θ=θ, q=q, f=f, ra=ra, rb=rb, v=v, z=z, M=M, w=w, 
        bond_spread=bond_spread, equity_spread=equity_spread, Y=Y,
     ξx=100*log.(x_series), ξB = 100*log.(B_series), ξα=100*log.(α_series), ξη=100*log.(η_series))
    return irf
end


struct Residual
    resids::Matrix{Float64}
end

Statistics.mean(r::Residual) = log10(mean(abs, r.resids))
Base.max(r::Residual) = log10(maximum(abs, r.resids))
max_E(r::Residual) = log10.(maximum(abs, r.resids, dims=2))[:]

function main()
    par = Params(para, σx=0.001, σB=1e-4, σσ=1e-4, σηa=1e-4  )
    m = Model(par, deg=2)
    coefs, solve_time = solve(m, tol=1e-7, damp=0.3, coefs_init=nothing)
    # simulate the model
    sim = Simulation(m, coefs, capT=50_000)
    #η_t = shock_transform.(p.η, exp.(sim.ξη))
    # check accuracy
    #tr = time(); 
    resids = Residuals(m, coefs, sim)
    # resids_time = time() - tr

    @show err_by_eq = max_E(resids)
    l1 = mean(resids)
    l∞ = max(resids)
    # tot_time = solve_time + simulation_time + resids_time
    # round3(x) = round(x, digits=3)
    # round2(x) = round(x, digits=2)
end

# Plot simulation
function simulation_plot(sim)
    fig, ax = subplots(1, 3, figsize=(20, 5))
    t = 250:1000
    ax[1].plot(t, sim.z[t], label="z")
    ax[1].plot(t, exp.(sim.ξx[t]), label="x")
    ax[1].plot(t, sim.w[t], label="w")
    ax[1].set_title("Subplot a: Productivity and wages")
    ax[1].legend()

    ax[2].plot(t, sim.f[t], label="f")
    ax[2].plot(t, sim.θ[t], label="θ")
    ax[2].set_title("Subplot b: Job finding rate and tightness")
    ax[2].legend()

    ax[3].plot(t, sim.u[t], label="u")
    ax[3].set_title("Subplot c: Unemployment")
    ax[3].legend()
    display(fig)
    PyPlot.savefig("simulations.pdf")
end

####################################################
" Impulse responses "

function impulse_response_plot(irf, shock_type="x")
    shock = zero(irf.ηx)
    if shock_type == "x"
        shock .= irf.ηx 
    elseif shock_type == "B"
        shock .= irf.ηB 
    elseif shock_type == "α"
        shock .= irf.ηα
    end
    shock_label = "$shock_type-shock"

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
        ax[1,1].plot(irf.θ, label="θ")
        ax[1,1].plot(irf.u, label="u")
        ax[1,1].legend(loc="best")
        ax[1,1].set_title("Market tightness and unemployment")
        
        ax[1,2].plot(irf.M, label="M")
        ax[1,2].plot(irf.Y, label="Y")
        ax[1,2].legend(loc="best")
        ax[1,2].set_title("Stock market capitalization and output")
        
        ax[2,1].plot(irf.bond_spread, label="Bond_spread")
        ax[2,1].plot(irf.equity_spread, label="Equity spread")
        ax[2,1].legend(loc="best")
        ax[2,1].set_title("Bond and equity liquidity premia")

        #ax[2, 2].plot(shock, label=shock_label)
        ax[2,2].plot(irf.z, label="Endogenous productivity")
        ax[2,2].legend(loc="best")
        tight_layout()
        display(fig)
        savefile = "irfs_"*"$shock_type"
        PyPlot.savefig("irfs.pdf")
end


# irf_x = impulse_response(m, coefs, scal=-1.0, shock_type="x")
# impulse_response_plot(irf_x, "x")

# irf_B = impulse_response(m, coefs, scal=-10.0, shock_type="B")
# impulse_response_plot(irf_B, "B")

# irf_α = impulse_response(m, coefs, scal=-10.0, shock_type="α")
# impulse_response_plot(irf_B, "α")



#solve_time, simulation_time, resids_time, coefs, sim, resids
