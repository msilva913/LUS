using PlotlyJS
using PyPlot
# to construct Sobol sequences
using Sobol
using BasisMatrices, QuantEcon, NLsolve, Roots
using MAT
using Parameters

# standard library components
using Printf
using Statistics, JSON3
using Random: seed!
using LinearAlgebra: diagm, cholesky, dot
using InteractiveUtils: versioninfo
using DelimitedFiles: writedlm

include("functions_LUS.jl")

json_string = read("params_calib.json", String)
para = JSON3.read(json_string)
δ, zbar, w0, w1, ϕ, ρ, α, B, σ, k, A, ξ, Bg, λ, η = para

function Params(;ρx=0.979, σx=0.007)
    return ParaCalib2(ρx=ρx, σx=σx)
end


function columns(M)
    # extract columns
    return (view(M, :, i) for i in 1:size(M, 2))
end


function grid_size(p, deg)
    Dict(1 =>20, 2 =>100, 3 => 500, 4 => 1000, 5 => 2000)[deg]
end

# returns covariance matrix for 6 shocks in the model
vcov(p) = diagm(0 => [p.σx^2])


#ss = SteadyState(p)


# Given an instance of Params and SteadyState, construct the grid for solving the model

function Grids(p, deg=2)
    m = grid_size(p, deg)
    σ = [p.σx]
    ρ = [p.ρx]
    u_min, u_max = 0.02, 0.3
    ub = [2 * p.σx / sqrt(1 - p.ρx^2), u_max]
    lb = -ub
    lb[2] = u_min

    # construct SobolSeq
    s = SobolSeq(length(ub), ub, lb)
   
    seq = zeros(m, length(lb))

    for i in 1:m
        seq[i, :] = next!(s)
    end

    η = seq[:, 1:1]
    u  = seq[:, 2]

    # decompose shocks
    ηx = η[:, 1]
    

    # Store grid
    X = [u log.(u) ηx]
    # complete polynomial
    X0_G = Dict(
        1 => complete_polynomial(X, 1),
        deg => complete_polynomial(X, deg)
    )

    ϵ_nodes, ω_nodes = qnwmonomial1(vcov(p))

    ηx1 = p.ρx.*ηx .+ ϵ_nodes[:, 1]'
  

    Grids = (ηx, u, X, X0_G, ϵ_nodes, ω_nodes, ηx1, u_min=u_min, u_max=u_max)
    return Grids
end

# Construct Model type, which has instance of Params, SteadyState, and Grids

function Model(p; deg=2)
    #p = Params()
    s = SteadyState(p)
    g = Grids(p, deg)
    Model = (p=p, s=s, g=g, deg=deg)
    return Model
end

Base.show(io::IO, m) = println(io, "LUS model")


# Helper function with takes control variables S, F, C and shocks δ, R, η_G, η_a, η_L, η_R and applied equilibrium conditions to
# produce   π, δ', Y, L, Y_n, R'
function Base.step(m, J, θ, u_lag, ηx)
    p, g = m.p, m.g
    # Helper function which updates quantities from J, u_lag
    @unpack zbar, σ, w1, δ, B, Bg, α, λ, ρ, η = m.p

    f = jf.(θ, A, ξ)
    q = vf.(θ, A, ξ)
    u = @. (1-f)*u_lag + δ*(1-u_lag)
    u = @.  min(u, g.u_max)
    u = @.  max(u, g.u_min)
    n =  @.(1 -u)
   
    ra = rafunc.(J, n, Ref(p))
    rb = rbfunc.(J, n, Ref(p))
  
    # Check bounds
   
    z = zfunc.(J, n, exp.(ηx), Ref(p))
    v = @. θ*u_lag
    
    return u, q, f, ra, rb, v, z
end


# Solution routine

function initial_coefs(m, deg)
    npol = size(m.g.X0_G[deg], 2)
    coefs = zeros(npol, 3)
    # initialize at steady-state values (non-constant terms are zero)
    coefs[1, :] = [m.s.J, m.s.θ, m.s.w]
    return coefs
end


function solve(m;damp=0.3, deg=2, tol=1e-7, verbose::Bool=true)
    # simplify notation
    @unpack zbar, σ, δ, B, Bg, α, λ, ρ, η, w0 = m.p
    β = 1.0/(1.0+ρ)
    p, g, deg = m.p, m.g, m.deg
    n, n_nodes = size(m.g.ηx, 1), length(m.g.ω_nodes)
    # number of policies to solve for
    npolicies = 3
    nstates = size(m.g.X)[2]
    ## allocate memory
    # euler equations
    e = zeros(n, npolicies)

    # previous policies
    J_old = ones(n)
    θ_old = ones(n)
    w_old = ones(n)

    # future policies
    J1 = zeros(n, n_nodes)
    θ1 = zeros(n, n_nodes)
    w1 = zeros(n, n_nodes)

    local coefs, start_time
    # set up matrices for this degree
    X0_G = g.X0_G[deg]

    # future basis matrix for policies
    X1 = Array{Float64}(undef, n, n_complete(nstates, deg))

    # initialize coefs
    coefs = initial_coefs(m, deg)

    error = 1.0
    it = 0

    start_time = time()
    # solve at this degree of complete polynomial
    while error > tol
        it += 1;
        # current choices (at t)
        # --------------------------------
        J = X0_G*coefs[:, 1]
        θ = X0_G*coefs[:, 2]
        w = X0_G*coefs[:, 3]
    
        out = step(m, J, θ, m.g.u, m.g.ηx)
        u0, q0, f0, ra0, rb0, v0, z0 = out

        for node in 1:n_nodes
            # Form complete polynomial of degree "Degree" at t+1 on future state 
            grid1 = hcat(u0, log.(u0), m.g.ηx1[:, node])
            complete_polynomial!(X1, grid1, deg)

            J1[:, node] = X1*coefs[:, 1]            # Compute E_{t+1}
            θ1[:, node] = X1*coefs[:, 2]
            w1[:, node] = X1*coefs[:, 3]

        end
        # Next-period quantities
        out = step(m, J1, θ1, u0, g.ηx1)
        u1, q1, f1, ra1, rb1, v1, z1 = out
        # Evaluate conditional expectations in the Euler equations
        #---------------------------------------------------------
        #(n x n_nodes)*(n_nodes x 1)
        # Use free entry condition to update tightness
        θ_new =  θ_fun.(J, 1.0 .-u0, Ref(p))
        z_new = (zfunc.(J, 1 .-u0, Ref(p)) + z0)./2
        J_new = @.(z_new - w +(1-δ)*J1/(1+ra1))*g.ω_nodes

        diff = 1.0
        #while diff > 1e-7
        out = step(m, J_new, θ_new, m.g.u, m.g.ηx)
        u0_new, q0, f0, ra0, rb0, v0, z0 = out
        u0_new = 0.9*u0 + 0.1*u0_new
        # Update another round
        θ_new .=  0.1*θ_fun.(J_new, 1.0 .-u0_new, Ref(p)) + 0.9*θ
        z_new .= 0.1*zfunc.(J_new, 1 .-u0_new, Ref(p)) + 0.9*z0
        J_new .= @.(z_new - w +(1-δ)*J1/(1+ra1))*g.ω_nodes
        J_new .= 0.9*J + 0.1*J_new
        #ra1 = rafunc.(J_new, 1.0 .-u0, Ref(p))
        @show diff = maximum(abs, 1.0 .- z_new./z0)
        #end

        # Update wage using wage equation
        w_new = @.((1-ϕ)*w0 + ϕ*(z0+β*(1+ra1)*k*θ) + ϕ*(1-δ)*k/q0*(1-β*(1+ra1)))*g.ω_nodes

        # Update J using Firm Bellman
       



        e[:, 1] .= J_new
        e[:, 2] .= θ_new
        e[:, 3] .= w_new
      
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
        error = mean(abs, 1.0.-J_new./J_old) +
                mean(abs, 1.0.-θ_new./θ_old) +
                mean(abs, 1.0.-w_new./w_old)

        if (it % 20 == 0) && verbose
            @printf "On iteration %d err is %6.7e\n" it error
        end

        # Store the obtained values for E_t on the grid to
        # be used on the subsequent iteration in Section 10.2.6
        #-----------------------------------------------------------------------
        copy!(J_old, J_new)
        copy!(θ_old, θ_new)
        copy!(w_old, w_new)
    end
    return coefs, time() - start_time
end


function Simulation(m, coefs::Matrix; capT=50_000, burn_in=500)
    # 11) Simulating a time sries Solution
    s = m.s
    T = capT + burn_in
    #----------------------------------------
    rands = randn(T, 1)
     # Initialize the values of 6 exogenous shocks
    #--------------------------------------------
    ηx = zeros(T)

    # Generate the series for shocks
    #-------------------------------
    @inbounds for t in 1:T-1
        ηx[t+1] = m.p.ρx*ηx[t] + m.p.σx*rands[t, 1]
    end

    u  = ones(T+1).*s.u # Time series of u
    θ  = Array{Float64}(undef, T)   # Time series of E(t)
    w = similar(θ)
    J = similar(θ)
    q = similar(θ)
    ra = similar(θ)
    rb = similar(θ)
    v = similar(θ)
    z = similar(θ)

    pol_bases = Array{Float64}(undef, 1, size(coefs, 1))
    # first period account accounts for lagged unemployment; proper time series starts at 2
    @inbounds for t in 2:(T)
        # construct the matrix of explanatory variables "pol_bases" on the series of state variables
        complete_polynomial!(
            pol_bases,
            # lagged unemployment and productivity
            hcat(u[t-1], log(u[t-1]), ηx[t]),
            m.deg
        )
        # Extract policy function
        policies = pol_bases*coefs
        J[t] = policies[1]
        θ[t] = policies[2]
        w[t] = policies[3]
        u[t], q[t], __, ra[t], rb[t], v[t], z[t] = step(m, J[t], θ[t], u[t-1], ηx[t])
    end

    # remove last element of u
    pop!(u)
    M = @. (1-u)*J
 
    out = [θ w u J q ra rb v z M][(burn_in+1):(end-1), :]
    θ, w, u, J, q, ra, rb, v, z, M = columns(out)
    Simulation = (θ=θ, w=w, u=u, J=J, q=q, ra=ra, rb=rb, v=v, z=z, M=M, ηx=ηx)
    return Simulation
end


function impulse_response(m, coefs; irf_length=48, scal=1.0)
    p = m.p
    ηx = zeros(irf_length)
    ηx[1] = p.σx*scal
    # Simulate shocks
    for t in 1:(irf_length-1)
        ηx[t+1] = p.ρx*ηx[t]
    end
    x = exp.(ηx)
    x_bas = ones(irf_length)
    pol_bases = Array{Float64}(undef, 1, size(coefs, 1))

    function impulse(x_series)
        u = zeros(irf_length+1)
        u[1] = m.s.u
        A = Array{Float64}(undef, irf_length, 10)
        θ, w, J, q, f, ra, rb, v, z, M = columns(A)
        for t in 2:irf_length
            complete_polynomial!(
                pol_bases,
                hcat(u[t-1], log(u[t-1]), log(x_series[t])),
                m.deg
            )
            # Extract policy function
            policies = pol_bases*coefs
            J[t] = policies[1]
            θ[t] = policies[2]
            w[t] = policies[3]
            out = step(m, J[t], θ[t], u[t-1], log(x_series[t]))
            u[t], q[t], f[t], ra[t], rb[t], v[t], z[t] = out
            M[t] = (1-u[t])*J[t]
        end
        pop!(u)
        out = [u q f ra rb v z M w][2:end, :]
        return out
    end

    out_imp = impulse(x)
    out_bas = impulse(x_bas)
    irf_res = @. 100*log(out_imp/out_bas)
    u, q, f, ra, rb, v, z, M, w = columns(irf_res)
    irf = (u=u, q=q, f=f, ra=ra, rb=rb, v=v, z=z, M=M, w=w, ηx=100*ηx)
    return irf
end


struct Residuals
    resids::Matrix{Float64}
end

function Residuals(m, coefs::Matrix, sim)
    p, g = m.p, m.g
    @unpack zbar, σ, w1, δ, B, Bg, α, λ, ρ, η, w0 = p
    β = 1/(1+ρ)
    capT = length(sim.w)
    resids = zeros(4, capT-1)

    # Integration method for evaluating accuracy
    # ------------------------------------------
    # Monomial integration rule with 2N^2+1 nodes
    ϵ_nodes, ω_nodes = qnwmonomial2(vcov(m.p))
    n_nodes = length(ω_nodes)

    # Allocate for arrays needed in the loop
    basis_mat = Array{Float64}(undef, n_nodes, 3)
    X1 = Array{Float64}(undef, n_nodes, size(coefs, 1))

    ηx1 = Array{Float64}(undef, n_nodes)

    for t in 1:(capT-1)                 # For each given point,
        # Take the corresponding value for shocks at t
        #---------------------------------------------
        ηx = sim.ηx[t]  # ηR(t)

        # Exctract time t values for all other variables (and t+1 for R, δ)
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
        copy!(ηx1, ηx*m.p.ρx .+ ϵ_nodes[:, 1])

        basis_mat[:, 1] .= u
        basis_mat[:, 2] .= log.(u)
        basis_mat[:, 3] .= ηx1

        # Future choices at t+1
        #----------------------
        # Form a complete polynomial of degree "Degree" (at t+1) on future state
        # variables; n_nodes-by-npol
        complete_polynomial!(X1, basis_mat, m.deg)

        # Compute next-period policies
        J1 = X1*coefs[:, 1]
        θ1 = X1*coefs[:, 2]
        out = step(m, J1, θ1, u, ηx1)
        u1, q1, __, ra1, rb1, v1, z1 = out
         
        # Compute residuals 
        # Revenue function
        resids[1, t] = 1 - zfunc(J, 1-u, exp(ηx), p)/z
        # Euler equation for firms
        resids[2, t] = 1 - dot(ω_nodes, @.(z -w +(1-δ)*J1/(1+ra1)))/J

        # Free entry condition
        resids[3, t] = 1.0 - θ_fun(J,1-u,p)/θ
        # Wage equation
        resids[4, t] = 1.0 - dot(ω_nodes, @.((1-ϕ)*w0 + ϕ*(z+β*(1+ra1)*k*θ) + ϕ*(1-δ)*k/q*(1-β*(1+ra1))))/w
    end
    # discard the first burn observations
    return Residuals(resids)
end

Statistics.mean(r::Residuals) = log10(mean(abs, r.resids))
Base.max(r::Residuals) = log10(maximum(abs, r.resids))
max_E(r::Residuals) = log10.(maximum(abs, r.resids, dims=2))[:]

# Running the combined
p = Params()
m = Model(p, deg=2)

coefs, solve_time = solve(m, tol=1e-8, damp=0.3)

# simulate the model
ts = time()
sim = Simulation(m, coefs)
simulation_time = time() - ts

# check accuracy
tr = time(); 
resids = Residuals(m, coefs, sim)
resids_time = time() - tr

err_by_eq = max_E(resids)
l1 = mean(resids)
l∞ = max(resids)
tot_time = solve_time + simulation_time + resids_time
round3(x) = round(x, digits=3)
round2(x) = round(x, digits=2)

# Plot simulation
fig, ax = subplots(1, 3, figsize=(20, 5))
t = 250:1000
ax[1].plot(t, exp.(ηx[t]), label="x")
ax[1].plot(t, sim.w1[t], label="w")
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

####################################################
" Impulse responses "
irf = impulse_response(m, coefs, scal=-1.0)

fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    #ax[1].plot(irf.θ, label="θ")
    #ax[1].plot(irf.v, label="v")
    ax[1].legend(loc="best")
    ax[1].set_title("Market tightness and vacancies")
    
    ax[2].plot(irf.u, label="u")
    ax[2].plot(irf.f, label="f")
    ax[2].legend(loc="best")
    ax[2].set_title("Unemployment and the job finding rate")
    
    ax[3].plot(irf.ηx, label="x")
    ax[3].plot(irf.w, label="w")
    ax[3].legend(loc="best")
    ax[3].set_title("Productivity shock and wages")
    tight_layout()
    display(fig)
    PyPlot.savefig("irfs.pdf")

#solve_time, simulation_time, resids_time, coefs, sim, resids














