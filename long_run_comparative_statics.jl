using PyPlot
using Parameters, CSV
using LinearAlgebra

include("steady_state_convexity.jl")

function comparative_statics(para, symbol, vals)
    θ_vals = zero(vals)
    u_vals = zero(vals)
    K_vals = similar(vals)
    int_out_vals = similar(vals)
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
        int_out_vals[i] = steady.int_out
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
    Y_vals = similar(vals)
    C_vals = similar(vals)

    Y_vals[1] = @. Y1_vals[1] + p_star*Y2_vals[1] 
    C_vals[1] = @. C1_vals[1] + p_star*C2_vals[1]
    for i in 2:length(vals)
        Y_Lasp = (Y1_vals[i] + p_vals[i-1]*Y2_vals[i])/(Y1_vals[i-1] + p_vals[i-1]*Y2_vals[i-1])
        Y_Paasche =(Y1_vals[i] + p_vals[i]*Y2_vals[i])/(Y1_vals[i-1] + p_vals[i]*Y2_vals[i-1])
        Y_growth = sqrt(Y_Lasp*Y_Paasche)

        C_Lasp = (C1_vals[i] + p_vals[i-1]*C2_vals[i])/(C1_vals[i-1] + p_vals[i-1]*C2_vals[i-1])
        C_Paasche =(C1_vals[i] + p_vals[i]*C2_vals[i])/(C1_vals[i-1] + p_vals[i]*C2_vals[i-1])
        C_growth = sqrt(C_Lasp*C_Paasche)

        Y_vals[i] = Y_vals[i-1]*Y_growth
        C_vals[i] = C_vals[i-1]*C_growth 
    end

    return [vals, θ_vals, u_vals, Y_vals, K_vals, bond_spread_vals, ra_vals, M_vals, p_vals, C_vals, C1_vals, C2_vals, z_vals]
end

function plot_statics(out, symbol; savefig = "", levels=true)
    vals, θ_vals, u_vals, Y_vals, K_vals, bond_spread_vals, ra_vals, M_vals, p_vals, C_vals, C1_vals, C2_vals, z_vals = out
    if levels
        vars =  [vals, θ_vals, u_vals, Y_vals, K_vals, bond_spread_vals, z_vals, M_vals, p_vals, C_vals, C1_vals, C2_vals]
    else
        vars = [100*log.(x/x[1]) for x in (vals, θ_vals, u_vals, Y_vals, K_vals, bond_spread_vals, z_vals, M_vals, p_vals,
                                     C_vals, C1_vals, C2_vals)]
    end
    labels = [symbol, :θ, :u, :Y, :K, :Bond_spread, :z_vals, :M, :p, :C, :C1, :C2]

    fig = plt.figure(figsize=(16, 8))
    for (i, key) in enumerate(vars)
        var = vars[i]
        lab = labels[i]
        ax = fig.add_subplot(3, 4, i)
        ax.plot(vals, var, linewidth=2)
        ax.set_title(lab)
        ax.set_xlabel(symbol)
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


json_string = read("params_calib.json", String)
par = JSON3.read(json_string)
s, δ, δ_k, Z, b, ϕ, η, ρ, α, σ, B, σ_1, σ_2, λ, γ, ω, A, ξ, Bg, η_a, η_k = par
para = ParaCalib(s=s, δ=δ, δ_k=δ_k, Z=Z, b=b, ϕ=ϕ, η=η, ρ=ρ, σ=σ, B=B, σ_1=σ_1, σ_2=σ_2, λ=λ,
    γ=γ, ω=ω, A=A, ξ=ξ, Bg=Bg, η_a=η_a, η_k=η_k)
ss = steady_state(para)


Bg_vals = 0:0.1:5
out = comparative_statics(para, :Bg, Bg_vals)
vals, θ_vals, u_vals, Y_vals, K_vals, bond_spread_vals, ra_vals, M_vals, p_vals, C_vals, C1_vals, C2_vals, z_vals = out
plot_statics(out, :Bg)

B_vals = 1:0.1:20
out = comparative_statics(para, :B, B_vals)
vals, θ_vals, u_vals, Y_vals, K_vals, bond_spread_vals, ra_vals, M_vals, p_vals, C_vals, C1_vals, C2_vals, z_vals = out
plot_statics(out, :B)

# η_k_vals = 0.0:0.05:0.4
# out = comparative_statics(para, :η_k, η_k_vals)
# vals, θ_vals, u_vals, Y_vals, K_vals, j_vals, ra_vals, M_vals, p_vals, C_vals, c2_vals, Liq_vals = out

σ_vals = 0.0001:0.005:0.1
out = comparative_statics(para, :σ, σ_vals)
vals, θ_vals, u_vals, Y_vals, K_vals, bond_spread_vals, ra_vals, M_vals, p_vals, C_vals, C1_vals, C2_vals, z_vals = out
plot_statics(out, :σ)

ηa_vals = 0.1:0.025:0.5
out = comparative_statics(para, :η_a, ηa_vals)
vals, θ_vals, u_vals, Y_vals, K_vals, bond_spread_vals, ra_vals, M_vals, p_vals, C_vals, C1_vals, C2_vals, z_vals = out
plot_statics(out, :η_a)

 Z_vals = 1:0.1:1.4
 out = comparative_statics(para, :Z, Z_vals)
 vals, θ_vals, u_vals, Y_vals, K_vals, bond_spread_vals, ra_vals, M_vals, p_vals, C_vals, C1_vals, C2_vals, z_vals = out
 plot_statics(out, :Z)
