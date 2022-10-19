using PlotlyJS
using PyPlot
# to construct Sobol sequences
using Sobol
using BasisMatrices, QuantEcon, NLsolve, Roots
using DataFrames
using MAT
using Parameters
using PyCall

# standard library components
using Printf
using Statistics, JSON3
using Random: seed!
using LinearAlgebra: diagm, cholesky, dot
using InteractiveUtils: versioninfo
using DelimitedFiles: writedlm
#cd("C:\\Users\\BIZtech\\Dropbox\\Unemployment Stocks II\\JEDC revision\\Programs\\Julia 25 July 2022")
#cd("C:\\Users\\TJSEM\\Dropbox\\Unemployment Stocks II\\JEDC revision\\Programs\\Julia 25 July 2022")
include("functions_LUS.jl")
include("time_series_fun.jl")

json_string = read("params_calib.json", String)
para = JSON3.read(json_string)
#δ, zbar, w0, w1, ϕ, ρ, α, B, σ, k, A, ξ, Bg, λ, η = para

function Params(;ρx=0.979, σx=0.007, ρB=0.979, σB=0.7, ρα=0.979, σα=0.7, ρη=0.979, ση=0.7)
    return ParaCalib2(ρx=ρx, σx=σx, ρB=ρB, σB=σB, ρα=ρα, σα=σα, ρη=ρη, ση=ση)
end


function columns(M)
    # extract columns
    return (view(M, :, i) for i in 1:size(M, 2))
end


function grid_size(p, deg)
    Dict(1 =>20, 2 =>200, 3 => 300, 4 => 1000, 5 => 2000)[deg]
end

# returns covariance matrix for 6 shocks in the model
vcov(p) = diagm(0 => [p.σx^2, p.σB^2, p.σα^2, p.ση^2])


#ss = SteadyState(p)


# Given an instance of Params and SteadyState, construct the grid for solving the model

function Grids(p, deg=2)
    m = grid_size(p, deg)
    σ = [p.σx, p.σB, p.σα, p.ση]
    ρ = [p.ρx, p.ρB, p.ρα, p.ρη]

    u_min, u_max = 0.025, 0.25
    ub = [2 * p.σx / sqrt(1 - p.ρx^2),
          2*p.σB/sqrt(1-p.ρB^2),
          2*p.σα/sqrt(1-p.ρα^2),
          2*p.ση/sqrt(1-p.ρη^2),
           u_max]
    lb = -ub
    lb[5] = u_min

    # construct SobolSeq
    s = SobolSeq(length(ub), ub, lb)
   
    seq = zeros(m, length(lb))

    for i in 1:m
        seq[i, :] = next!(s)
    end
    #exogenous states
    ξ = seq[:, 1:4]
    # endogenous states
    u  = seq[:, 5]

    # decompose shocks
    ξx = ξ[:, 1]
    ξB = ξ[:, 2]
    ξα = ξ[:, 3]
    ξη = ξ[:, 4]

    # Store grid
    X = [u log.(u) ξx ξB ξα ξη]
    # complete polynomial
    X0_G = Dict(
        1 => complete_polynomial(X, 1),
        deg => complete_polynomial(X, deg)
    )

    ϵ_nodes, ω_nodes = qnwmonomial1(vcov(p))

    ξx1 = p.ρx.*ξx .+ ϵ_nodes[:, 1]'
    ξB1 = p.ρB.*ξB .+ ϵ_nodes[:, 2]'
    ξα1 = p.ρα.*ξα .+  ϵ_nodes[:, 3]'
    ξη1 = p.ρη.*ξη .+ ϵ_nodes[:, 4]'

    Grids = (ξx, ξB, ξα, ξη, u, X, X0_G, ϵ_nodes, ω_nodes, 
    ξx1, ξB1, ξα1, ξη1, u_min=u_min, u_max=u_max)
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
function Base.step(m, θ, u_lag, ξx, ξB, ξα, ξη)
    p, g = m.p, m.g
    # Helper function which updates quantities from J, u_lag
    @unpack zbar, σ, δ, B, Bg, α, λ, ρ, η, A, ξ = m.p

    x_shock = exp.(ξx)
    B_shock= exp.(ξB)
    α_shock = exp.(ξα)
    η_shock = exp.(ξη)

    f = jf.(θ, A, ξ)
    q = vf.(θ, A, ξ)
    u = @. (1-f)*u_lag + δ*(1-u_lag)
    u = @.  min(u, g.u_max)
    u = @.  max(u, g.u_min)
    n =  @.(1 -u)

    if typeof(θ) == Float64
        J = fzero(x -> (1.0-θ_fun(x, n, m.p, B_shock=B_shock, α_shock=α_shock, η_shock=η_shock)/θ)*100, m.s.J)
    else
        J = [fzero(x -> (1.0-θ_fun(x, n_val, m.p, B_shock=B_val, α_shock=α_val, η_shock=η_val)/θ_val)*100, m.s.J) 
            for (θ_val, n_val, B_val, α_val, η_val) in zip(θ, n, B_shock, α_shock, η_shock)]
    end

    ra = rafunc.(J, n, Ref(p), B_shock, α_shock, η_shock)
    rb = rbfunc.(J, n, Ref(p), B_shock, α_shock, η_shock)

    z = zfunc.(J, n, Ref(p), x_shock, B_shock, α_shock, η_shock)
    v = @. θ*u_lag
    # Check bounds
   
    return u, J, q, f, ra, rb, v, z
end


# Solution routine

function initial_coefs(m, deg)
    npol = size(m.g.X0_G[deg], 2)
    coefs = zeros(npol, 2)
    # initialize at steady-state values (non-constant terms are zero)
    coefs[1, :] = [sqrt(m.s.θ), m.s.w]
    return coefs
end


function solve(m;damp=0.3, deg=2, tol=1e-7, verbose::Bool=true)
    # simplify notation
    @unpack zbar, σ, δ, B, Bg, α, λ, ρ, η, w0, A, ξ, ϕ, k = m.p
    β = 1.0/(1.0+ρ)

    p, g, deg = m.p, m.g, m.deg
    n, n_nodes = size(m.g.ξx, 1), length(m.g.ω_nodes)
    # number of policies to solve for
    npolicies = 2
    nstates = size(m.g.X)[2]
    ## allocate memory
    # euler equations
    e = zeros(n, npolicies)

    # previous policies
    θ_old = ones(n)
    w_old = ones(n)

    # future policies
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
        θ = (X0_G*coefs[:, 1]).^2
        w = X0_G*coefs[:, 2]
    
        out = step(m, θ, g.u, g.ξx, g.ξB, g.ξα, g.ξη)
        u0, J0, q0, f0, ra0, rb0, v0, z0 = out

        for node in 1:n_nodes
            # Form complete polynomial of degree "Degree" at t+1 on future state 
            grid1 = hcat(u0, log.(u0), g.ξx1[:, node], g.ξB1[:, node], g.ξα1[:, node], g.ξη1[:, node])
            complete_polynomial!(X1, grid1, deg)

            θ1[:, node] = (X1*coefs[:, 1]).^2
            #w1[:, node] = X1*coefs[:, 2]

        end
        # Next-period quantities
        out = step(m, θ1, u0, g.ξx1, g.ξB1, g.ξα1, g.ξη1)
        u1, J1, q1, f1, ra1, rb1, v1, z1 = out
        # Evaluate conditional expectations in the Euler equations
        #---------------------------------------------------------
        #(n x n_nodes)*(n_nodes x 1)
        # Calculate J using Euler equation and update theta
        # z_new = zfunc.(J, 1. .-u0, Ref(p))
        # z_new = (z_new + z0)./2
        w_new = @.((1-ϕ)*w0 + ϕ*(z0+β*(1+ra1)*k*θ) + ϕ*(1-δ)*k/q0*(1-β*(1+ra1)))*g.ω_nodes
        J_new = @.(z0 - w_new +(1-δ)*J1/(1+ra1))*g.ω_nodes
        θ_new =  θ_fun.(J_new, 1.0 .-u0, Ref(p), exp.(g.ξB), exp.(g.ξα), exp.(g.ξη))

        # Update wage using wage equation
        # Update J using Firm Bellman
       
        e[:, 1] .= sqrt.(θ_new)
        e[:, 2] .= w_new
      
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
        error = maximum(abs, 1.0.-θ_new./θ_old) +
                maximum(abs, 1.0.-w_new./w_old)

        if (it % 20 == 0) && verbose
            @printf "On iteration %d err is %6.7e\n" it error
        end

        # Store the obtained values for E_t on the grid to
        # be used on the subsequent iteration in Section 10.2.6
        #-----------------------------------------------------------------------
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
    rands = randn(T, 4)
     # Initialize the values of 6 exogenous shocks
    #--------------------------------------------
    ξx = zeros(T)
    ξB = zeros(T)
    ξα = zeros(T)
    ξη = zeros(T)

    # Generate the series for shocks
    #-------------------------------
    @inbounds for t in 1:T-1
        ξx[t+1] = m.p.ρx*ξx[t] + m.p.σx*rands[t, 1]
        ξB[t+1] = m.p.ρB*ξB[t] + m.p.σB*rands[t, 2]
        ξα[t+1] = m.p.ρα*ξα[t] + m.p.σα*rands[t, 3]
        ξη[t+1] = m.p.ρη*ξη[t] + m.p.ση*rands[t, 4]
    end

    u  = ones(T).*s.u # Time series of u
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
            hcat(u[t-1], log(u[t-1]), ξx[t], ξB[t], ξα[t], ξη[t]),
            m.deg
        )
        # Extract policy function
        policies = pol_bases*coefs
        θ[t], w[t] = policies[1]^2, policies[2]
        out = step(m, θ[t], u[t-1], ξx[t], ξB[t], ξα[t], ξη[t])
        u[t], J[t], q[t], __, ra[t], rb[t], v[t], z[t] = out
    end

    # remove last element of u
    #pop!(u)
    M = @. (1-u)*J
    f = @.θ*q
    y = @. (1-u)*z
    out = [θ w u J f q ra rb v z M y ξx ξB ξα ξη][(burn_in+1):(end), :]
    θ, w, u, J, f, q, ra, rb, v, z, M, y, ξx, ξB, ξα, ξη = columns(out)
    Simulation = (θ=θ, w=w, u=u, J=J, f=f, q=q, ra=ra, rb=rb, v=v, z=z, M=M, y=y,
                    ξx=ξx, ξB=ξB, ξα=ξα, ξη=ξη)
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
        θ1 = (X1*coefs[:, 1]).^2
        out = step(m, θ1, u, ξx1, ξB1, ξα1, ξη1)
        u1, J1, q1, __, ra1, rb1, v1, z1 = out
         
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
            θ[t], w[t] = pol_bases*coefs
            out = step(m, θ[t], u[t-1], log(x_series[t]), log(B_series[t]), log(α_series[t]), log(η_series[t]))
            u[t], J[t], q[t], __, ra[t], rb[t], v[t], z[t] = out
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

# Running the combined
p = Params(σB=0.03, σα=0.03, ση=0.03)
m = Model(p, deg=2)

coefs, solve_time = solve(m, tol=1e-8, damp=0.3)

# simulate the model
ts = time()
sim = Simulation(m, coefs)
simulation_time = time() - ts


#η_t = shock_transform.(p.η, exp.(sim.ξη))
# check accuracy
tr = time(); 
# resids = Residuals(m, coefs, sim)
# resids_time = time() - tr

# @show err_by_eq = max_E(resids)
# l1 = mean(resids)
# l∞ = max(resids)
# tot_time = solve_time + simulation_time + resids_time
# round3(x) = round(x, digits=3)
# round2(x) = round(x, digits=2)
###############################################################
# moments
#dat = DataFrame([sim.M sim.u sim.z], [:M :u :z])

# log deviations from mean of stationary distribution 
" Log deviations from stationary mean "
out = [log.(getfield(sim, x)./mean(getfield(sim,x))) for x in [:y, :u, :M]]
#data = DataFrames.DataFrame(vars, :auto).*100
#data = convert(DataFrames.DataFrame, vars)
df = DataFrames.DataFrame(out', :auto)
rename!(df, [:y, :u, :M])

# start = Date(1, 1, 1)
# finish = start + Month(length(sim.u)-1)
# dates = start:Month(1):finish
# #df[:date] = dates


df = Pandas.DataFrame(data)
#index(df) = Pandas.period_range(start="01-01-01", periods=50_000, freq="M")

py"""
import numpy as np
import pandas as pd
df = $df
#dates = pd.date_range(start="01-01-01", periods=500000, freq="M")
#df = pd.DataFrame(df, index=dates)
#df.index = dates
#df_q = df.resample("Q").mean().dropna()

def monthly_to_quarterly_datetime(sim, N=10000):
    
    capT= sim.shape[0]
    K = int(capT/N)
    date = pd.date_range(start='01-01-01', periods=K, freq='MS')
    df_array = pd.DataFrame()
    for i in range(N):
        df = sim.iloc[K*i:K*(i+1), :]
        df.index = date
        df = df.resample('QS').mean()
        df_array = pd.concat([df_array, df])
    
    return df_array

df_q = monthly_to_quarterly_datetime(df)
df_q.columns = ["y", "u", "M"]
"""

df_q = py"df_q"

function pd_to_df(df_pd)
    df = DataFrames.DataFrame()
    for col in [:y, :u, :M]
        df[!, col] = getproperty(df_pd, col).values
    end
    return df
end

df_q = pd_to_df(df_q)

# Apply bk filter
cycle = mapcols(col -> bkfilter(col, wl=6, wu=200, K=12), df_q)

#mom_data = moments(df_q, :M, [:u], var_names=[:M, :z, :u])
@show mom_data = moments(cycle, :y, [:u], var_names=[:y, :u, :M])
print(mom_data)


# Plot simulation
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


irf_x = impulse_response(m, coefs, scal=-1.0, shock_type="x")
impulse_response_plot(irf_x, "x")

irf_B = impulse_response(m, coefs, scal=-10.0, shock_type="B")
impulse_response_plot(irf_B, "B")

irf_α = impulse_response(m, coefs, scal=-10.0, shock_type="α")
impulse_response_plot(irf_B, "α")



#solve_time, simulation_time, resids_time, coefs, sim, resids














