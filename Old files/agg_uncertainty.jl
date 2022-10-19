
using PyPlot
using Plots
using BenchmarkTools
using LaTeXStrings, KernelDensity
using Parameters, CSV, Statistics, Random, QuantEcon
using NLsolve, Dierckx, Distributions, ArgParse
using LinearAlgebra, Roots, Optim, LinearInterpolations, Interpolations
using Printf
using Dierckx
using DataFrames, Pandas, JSON3
include("time_series_fun.jl")
include("functions_LUS.jl")

# Load calibrated parameters
json_string = read("params_calib.json", String)
para = JSON3.read(json_string)
δ, zbar, w0, w1, ϕ, ρ, α, B, σ, k, A, ξ, Bg, λ, η = para

# Calculate steady state
para = ParaCalib2(δ=δ,  w0=w0, ϕ=ϕ, ρ=ρ, α=α, B=B, σ=σ, k=k, A=A, ξ=ξ, Bg=Bg, λ=λ, η=η)
ss = steady_state(0, para)
@unpack n, J, spread_bond, spread_equity, r_a, r_b, w, z, M, w, z, θ = ss


NumPara = @with_kw ( 
    u_low = 0.025,
    u_high = 0.40,
    sig = 1e-6,
    max_iter = 1000,
    NU = 50,
    NS = 20,
    T = 1e5,
    ρ_x = 0.974,
    σ_x = 0.009,
    mc = rouwenhorst(NS, ρ_x, σ_x, 0),
    P = mc.p,
    z_vals = exp.(mc.state_values)
)


function update_params(self, cal)
    @unpack α, β, δ, θ = cal
    self.α = α
    self.β = β
    self.δ = δ
    self.θ = θ
    nothing
end



function RHS_fun_cons(l_pol::Function, para::Para)
    @unpack α, β, δ, θ, P, NK, NS, k_grid, A = para
    # consumption given state and labor
    
    RHS = zeros(NK, NS)
    @inbounds Threads.@threads for (i, k) in collect(enumerate(k_grid))
    #for (i, k) in enumerate(k_grid)
        for z in 1:NS
            # labor policy
            l_i = l_pol(k, z)
            # consumption and output
            y = A[z]*k^α*l_i^(1-α)
            c = (1-l_i)/θ*(1-α)*y/l_i
            #c = min(c, y)
            # update capital
            k_p = (1-δ)*k + A[z]*k^α*l_i^(1-α) - c
            for z_hat in 1:NS
                # update labor supply via interpolation
                l_p = l_pol(k_p, z_hat)
                # update consumption
                y_p = A[z_hat]*k_p^α*l_p^(1-α)
                c_p = (1-l_p)/θ*(1-α)*y_p/l_p
                #c_p = min(c_p, y_p)
                # future marginal utility
                RHS[i, z] += P[z, z_hat]*((1/c_p)*(α*y_p/k_p+1-δ))
            end
        end
    end
    RHS = (β.*RHS)
    rhs_fun(k, z) = LinearInterpolation(k_grid, RHS[:, z], extrapolation_bc=Line())(k)
    return rhs_fun
end


function labor_supply_loss(l_i, k, z, RHS_fun,  para::Para)
    
    @unpack A, α, θ = para
    # convert domain from (0, infty) to (0, 1)
    # optimal consumption
    y = A[z]*k^α*l_i^(1-α)
    c = (1-l_i)/θ*(1-α)*y/l_i
    #c = min(c, y)
    error =  1/c - RHS_fun(k, z)
    return error
end


function solve_model_time_iter(l, para::Para; tol=1e-7, max_iter=1000, verbose=true, 
                                print_skip=25)
    # Set up loop 
    @unpack k_grid, NS, A, α, θ= para
    # Initial consumption level
    c = similar(l)
    c_new = similar(l)
    for z in 1:NS
        c[:, z] = A[z].*k_grid.^α.*l[:, z].^(1-α)
    end

    err = 1
    iter = 1
    while (iter < max_iter) && (err > tol)
        # interpolate given labor grid l
        l_pol(k, z) = Interpolate(k_grid, @view(l[:, z]), extrapolate=:reflect)(k)
        RHS_fun = RHS_fun_cons(l_pol, para)
        for (i, k) in enumerate(k_grid)
        #@inbounds Threads.@threads for (i, k) in collect(enumerate(k_grid))
            for z in 1:NS
                # solve for labor supply
                l_i = find_zero(l_i -> labor_supply_loss(l_i, k, z, RHS_fun, para), (1e-10, 0.99), Bisection() )
                #l_i = abs(h_i)/(1+abs(h_i))
                l[i, z] = l_i
                # implied consumption
                y = A[z]*k^α*l_i^(1-α)
                c_new[i, z] = (1-l_i)/θ*(1-α)*y/l_i
            end
        end
        #@printf(" %.2f", (mean(c)))
        err = maximum(abs.(c_new-c)/max.(abs.(c), 1e-10))
        if verbose && iter % print_skip == 0
            print("Error at iteration $iter is $err.")
        end
        iter += 1
        c = c_new
    
    end

    # Get convergence level
    if iter == max_iter
        print("Failed to converge!")
    end

    if verbose && (iter < max_iter)
        print("Converged in $iter iterations")
    end
    y = similar(c)
    inv = similar(c)
    for (i, k) in enumerate(k_grid)
        for z in 1:NS
            y[i, z] = A[z]*k^α*l[i, z]^(1-α)
        end
    end

    inv = y - c
    w = (1-α).*y./l
    R = α.*y./k_grid
    l_pol(k, z) = LinearInterpolation(k_grid, l[:, z], extrapolation_bc=Line())(k)
    return l, l_pol, c, y, inv, w, R
end


function simulate_series(l_mat::Array, para::Para, burn_in=200, capT=10000)

    @unpack ρ_x, σ_x, P, mc, A, α, θ, δ, k_grid = para
    l_pol(k, z) = Interpolate(k_grid, @view(l_mat[:, z]), extrapolate=:reflect)(k)
    capT = capT + burn_in + 1

    # Extract indices of simualtes shocks
    z_indices = simulate_indices(mc, capT)
    z_series = A[z_indices]
    # Simulate shocks
    
    k = ones(capT+1)
    var = ones(capT, 3)
    l, y, c = [ var[:,i] for i in 1:size(var, 2) ]

    for t in 1:capT
        l[t] = l_pol(k[t], z_indices[t])
        y[t] = z_series[t]*k[t]^α*l[t]^(1-α)
        c[t] = (1-l[t])/θ*(1-α)*y[t]/l[t]
        k[t+1] = (1-δ)*k[t] + y[t] - c[t]
    end
    k = k[1:(end-1)]
    i = y - c
    w = (1-α).*y./l
    R = α.*y./k
    lab_prod = y./l
    Simulation = (l=l, y=y, c=c, k=k, i=i, w=w, R=R,
                 lab_prod=lab_prod, η_x=log.(z_series), z_indices)
    return Simulation
end


function impulse_response(l_mat, para, k_init; irf_length=40, scale=1.0)

    @unpack ρ_x, σ_x, P, mc, A, α, θ, δ, k_grid = para

    # Bivariate interpolation (AR(1) shocks, so productivity can go off grid)
    L = Spline2D(k_grid, A, l_mat)

    η_x = zeros(irf_length)
    η_x[1] = σ_x*scale
    for t in 1:(irf_length-1)
        η_x[t+1] = ρ_x*η_x[t]
    end
    z = exp.(η_x)
    z_bas = ones(irf_length)

    function impulse(z_series)

        k = zeros(irf_length+1)
        l = zeros(irf_length)
        c = zeros(irf_length)
        y = zeros(irf_length)

        k[1] = k_init

        for t in 1:irf_length
            # labor
            l[t] = L(k[t], z_series[t])
            y[t] = z_series[t]*k[t]^α*l[t]^(1-α)
            c[t] = (1-l[t])/θ*(1-α)*y[t]/l[t]
            k[t+1] = (1-δ)*k[t] + y[t] - c[t]
        end

        k = k[1:(end-1)]
        i = y - c
        w = (1-α).*y./l
        R = α.*y./k
        lab_prod = y./l
        out = [c k l i w R y lab_prod]
        return out
    end

    out_imp = impulse(z)
    out_bas = impulse(z_bas)

    irf_res = similar(out_imp)
    @. irf_res = 100*log(out_imp/out_bas)
    #out = [log.(x./mean(getfield(simul, field))) for (x, field) in
    #zip([c, k[1:(end-1)], l, i, w, R, y, lab_prod], [:c, :k, :l, :i, :w, :R, :y, :lab_prod])]
    c, k, l, i, w, R, y, lab_prod = [irf_res[:, i] for i in 1:size(irf_res, 2)]

    irf = (l=l, y=y, c=c, k=k, i=i, w=w, R=R,
                 lab_prod=lab_prod, η_x=100*log.(z))
    return irf
end


function residual(l_pol, simul, para::Para; burn_in=200)
    capT = size(simul.c)[1]
    @unpack A, α, θ, P = para
    @unpack k, z_indices = simul

    " Pre-allocate arrays "
    #rhs_fun = RHS_fun_cons(l_pol, para)
    rhs_fun = RHS_fun_cons(l_pol, para)

    " Right-hand side of Euler equation "
    #rhs = RHS_fun.(k, z_indices)
    rhs = rhs_fun.(k, z_indices)
    loss = 1.0 .- simul.c .* rhs
    return loss[burn_in:end]
end  
   

para = Para()
cal = calibrate()
update_params(para, cal)
@unpack NK, NS, A, k_grid, α = para

# Initialize labor supply
l = ones(NK, NS)*0.33
# iterate until policy function converges
@code_warntype solve_model_time_iter(l, para)
l_mat, l_pol, c, y, inv, w, R = solve_model_time_iter(l, para, verbose=false)
#@btime solve_model_time_iter($l, $para)

" Solve for steady state "
steady = steady_state(cal)

" Simulation "
simul = simulate_series(l_mat, para) 
@unpack c, k, l, i, w, R, y, lab_prod, η_x, z_indices = simul

" Log deviations from stationary mean "
out = [log.(getfield(simul, x)./mean(getfield(simul,x))) for x in keys(steady)]
l, c, k, y, i, w, R, lab_prod = out
simul_dat = DataFrames.DataFrame(l=l, c=c, k=k, y=y, i=i, w=w, R=R, lab_prod=lab_prod)


fig, ax = subplots(1, 3, figsize=(20, 5))
t = 250:1000
ax[1].plot(t, c[t], label="c")
ax[1].plot(t, l[t], label="l")
ax[1].plot(t, i[t], label="i")
ax[1].plot(t, y[t], label="y")
ax[1].set_title("Consumption, investment, output, and labor supply")
ax[1].legend()

ax[2].plot(t, w[t], label="w")
ax[2].plot(t, R[t], label="R")
ax[2].set_title("Wage and rental rate of capital")
ax[2].legend()

ax[3].plot(t, η_x[t], label="x")
ax[3].plot(t, lab_prod[t], label="labor productivity")
ax[3].set_title("Total factor and labor productivity")
ax[3].legend()
display(fig)
PyPlot.savefig("simulations.pdf")

" Residuals "
res = residual(l_pol, simul, para)

" Impulse responses "
k_1 = mean(simul.k)
irf = impulse_response(l_mat, para, k_1, irf_length=60)

fig, ax = subplots(1, 3, figsize=(20, 5))
ax[1].plot(irf.c, label="c")
ax[1].plot(irf.i, label="i")
ax[1].plot(irf.l, label="l")
ax[1].plot(irf.y, label="y")
ax[1].set_title("Consumption, investmnt, output, and labor supply")
ax[1].legend()

ax[2].plot(irf.w, label="w")
ax[2].plot(irf.R, label="R")
ax[2].set_title("Wage and rental rate of capital")
ax[2].legend()

ax[3].plot(irf.η_x, label="x")
ax[3].plot(irf.lab_prod, label="Labor productivity")
ax[3].set_title("Total factor and labor productivity")
ax[3].legend()
display(fig)
PyPlot.savefig("rbc_irf.pdf")

" Moments from simulated data "
# convert to Pandas DataFrame
#simul_dat = Pandas.DataFrame(simul_dat)
mom = moments(simul_dat)
      



