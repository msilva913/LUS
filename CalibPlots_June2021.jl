
using PyPlot
using LaTeXStrings, KernelDensity
using Parameters, CSV, StatsBase, Statistics, Random
using NLsolve, Dierckx, Distributions, ArgParse
using LinearAlgebra, QuadGK, Roots, Optim


include("functions_LUS.jl")

### Plot the contour plots

para = ParaCalib2()
# check steady state
#fp(0, para)

steps =200
n_lims = [0.01,1.4]
J_lims2 = [0.001, 10.0]
revJdata = revJnull(n_lims, steps, 0, para)
ndata = nnull(J_lims2, steps, para)

XTrev = range(n_lims[1], stop = n_lims[2], length=steps)|>collect
XT2 = range(J_lims2[1], stop = J_lims2[2], length = steps)|>collect


revJdataPS = revJnull(n_lims, steps, 1,ParaCalib2(α=0.04,λ=0.1, zbar=0.8,  k=1.0, w1=1.0, Ag=1.25,γ=1.0, B=18.0, σ=.2))
ndataPS = nnull(J_lims2, steps, ParaCalib2(α=0.04,λ=0.1, zbar=0.8, δ=.035, ξ=.45, k=1.0, w1=1.0, Ag=1.25,γ=1.0, B=18.0, σ=.2))


# Calibrated contour plot

(figT, ax) = subplots(1, 1, figsize = (10, 10))

#ax = ax_bin[1,1]
ax.plot(XTrev,revJdata, color = (0.0, 0.2, 0.5), linewidth = 2.0)
ax.plot(ndata,XT2,color=(.541, .2, .141),linewidth=2.0)
ax.axis([-.01, 1.4, -.5, 10.0])
ax.set_xlabel(L"n", fontsize=12)
ax.set_ylabel(L"J", fontsize=12)
ax.legend([L"J=f(n,J)",L"n=g(n,J)"], fontsize = 12, frameon = 0, loc = "lower right")
display(figT)
PyPlot.savefig("/Users/billbranch/Dropbox/Unemployment Stocks II/latest draft/figs/calib_contour.pdf")
close(figT)
# Perfect storm contour plot

(figTb, axb) = subplots(1, 1, figsize = (10, 10))
#ax = ax_binb[2,1]
axb.plot(XTrev, revJdataPS, color = (0.0, 0.2, 0.5), linewidth = 2.0)
axb.plot(ndataPS, XT2,color=(.541, .2, .141),linewidth=2.0)
axb.axis([0.01, 1.4, -.5,2.0])
axb.set_xlabel(L"n", fontsize=12)
axb.set_ylabel(L"J", fontsize=12)
axb.legend([L"J=f(n,J)",L"n=g(n,J)"], fontsize = 12, frameon = 0, loc = "lower right")

display(figTb)
#savefig("/Users/billbranch/Dropbox/Unemployment Stocks II/latest draft/figs/ps_contour.png")
close(figTb)


# Dynamics

function dynmap(J0::Float64, n0::Float64, sim_length::Int64, FixFlag::Int64, para::ParaCalib2)

    J1, n1, J2, n2 = fp(FixFlag, para)

    

    Jlist = Array{Float64}(undef, sim_length)
    nlist = similar(Jlist)

    Jlist[1] = J0
    nlist[1] = n0

    for t in 2:sim_length

        function obj(x::Array{Float64,1})

            obj1 = Jfunc(nlist[t-1], x[1], x[2], FixFlag, para)-Jlist[t-1]
            obj2 = nfunc(nlist[t-1], x[1], para) - x[2]

            return [obj1 obj2]
        end

        sol1 = nlsolve(obj, [Jlist[t-1], nlist[t-1]])

        Jlist[t], nlist[t] = sol1.zero

    end

    return Jlist, nlist

end

function perturbEq(Je::Float64, ne::Float64, FixFlag::Int64, para_perturb::ParaCalib2, para::ParaCalib2)

    J1, n1, J2, n2 = fp(FixFlag, para)

    out1 = tempeq(Je, ne, n1, FixFlag, para_perturb)
    out2 = nfunc(n1, out1, para_perturb)

    return out1, out2

end

# generates J_t = tempeq (J^e_{t+1}, n^e_{t+1}, n_{t-1})
# then n_t = nfunc(J_t,n_{t-1})

function tempeq(Je::Float64, ne::Float64, n::Float64, FixFlag::Int64, para::ParaCalib2)

    function obj(x::Float64)

       return Jfunc(nfunc(n, x, para), Je, ne, FixFlag,para)-x

    end

    return fzero(x -> obj(x), Je)

end

function simulate_learning(ρ::Float64, gn::Float64, shock::Float64, type_shock::Int64, sim_length::Int64, FixFlag::Int64, para::ParaCalib2, para_perturb::ParaCalib2)

    # shock = percentage disturbance to parameter
    # type_shock = 1: B; 2: α; 3: Ag
    # gn = geometric discounting in expectations

    Jsim = Array{Float64}(undef, sim_length)
    nsim = similar(Jsim)
    Msim = similar(Jsim)
    Usim = similar(Jsim)
    Je = similar(Jsim)
    ne = similar(Jsim)
    Mchng = similar(Jsim)
    Uchng = similar(Jsim)

    J1, n1, J2, n2 = fp(FixFlag, para)

    Msim1 = J1*n1
    Usim1 = (1-n1)

    Je[1] = J1
    ne[1] = n1
    

    Bt = Array{Float64}(undef, sim_length)
    αt = similar(Bt)
    Agt = similar(Bt)

    @unpack B, α, σ, Ag = para

    αT = α
    σT = σ
    AgT = Ag

    if type_shock == 1
        Bt[1] = B
        Bt[2] = B*(1+shock)
        αt = fill(αT, sim_length)
        Agt = fill(AgT, sim_length)
    end

    if type_shock == 2
        Bt = fill(B, sim_length)
        αt[1] = α
        αt[2] = α*(1+shock)
        Agt = fill(AgT, sim_length)
    end

    if type_shock == 3
        Bt = fill(B, sim_length)
        αt = fill(αT, sim_length)
        Agt[1] = AgT
        Agt[2] = AgT*(1+shock)
    end

    Jsim[1] = J1
    nsim[1] = n1
    Msim[1] = Jsim[1]*nsim[1]
    Usim[1] = 1-n1
    Jsim[2] = 1.0*tempeq(Je[1], ne[1], n1, FixFlag, ParaCalib2(α = αt[2], σ = σT, Ag = Agt[2], B=Bt[2]))
    nsim[2] = nfunc(n1, Jsim[2], ParaCalib2(α = αt[2], σ = σT, Ag = Agt[2], B=Bt[2]))
    Mchng[1] = 0.0
    Uchng[1] = 0.0
    
    Je[2] = Je[1] + gn *( Jsim[1]-Je[1])
    ne[2] = ne[1] + gn *( nsim[1]-ne[1])
    Msim[2] = Jsim[2]*nsim[2]
    Usim[2] = 1-nsim[2]
    
    Mchng[2] = 1200*(log(Msim[2])-log(Msim1))  # percent change p/a
    Uchng[2] = 1200*(log(Usim[2])-log(Usim1))

    

    for t in 3:sim_length

        Je[t] = Je[t-1] + gn * (Jsim[t-1] - Je[t-1])
        ne[t] = ne[t-1] + gn * (nsim[t-1] - ne[t-1])

        Bt[t] = B+ρ*(Bt[t-1]-B)
        αt[t] = αT + ρ*(αt[t-1]-αT)
        Agt[t] = AgT + ρ*(Agt[t-1]-AgT)

        Jsim[t] = tempeq(Je[t], ne[t], nsim[t-1], FixFlag, ParaCalib2(α = αt[t], σ = σT, Ag = Agt[t], B = Bt[t]))
        nsim[t] = nfunc(nsim[t-1], Jsim[t], ParaCalib2(α = αt[t], σ = σT, Ag = Agt[t], B = Bt[t]))
        Msim[t] = Jsim[t]*nsim[t]
        Usim[t] = 1-nsim[t]
        Mchng[t] = 1200.0*(log(Msim[t])-log(Msim1))
        Uchng[t] = 1200*(log(Usim[t])-log(Usim1))

    end
    
    return Jsim, nsim, Msim, Usim, Je, ne, Bt, Mchng, Uchng

end


        


# demand shocks

sim_length = 48

Jsim1, nsim1, Msim1, Usim1, Jesim1, nesim1, Bsim1, Mc1, Uc1 = simulate_learning(0.9,.01, -0.27, 1, sim_length, 0, ParaCalib2(), ParaCalib2())
Jsim2, nsim2, Msim2, Usim2, Jesim2, nesim2, Bsim2, Mc2, Uc2 = simulate_learning(0.9,.17, -0.27, 1, sim_length, 0, ParaCalib2(), ParaCalib2())
Jsim3, nsim3, Msim3, Usim3, Jesim3, nesim3, Bsim3, Mc3, Uc3 = simulate_learning(0.9,.50, -.27, 1, sim_length, 0, ParaCalib2(), ParaCalib2())
Jsim4, nsim4, Msim4, Usim4, Jesim4, nesim4, Bsim4, Mc4, Uc4 = simulate_learning(0.9,.20, -0.27, 1, sim_length, 0, ParaCalib2(), ParaCalib2())

Js0, ns0, Ms0, Us0, Jes0, nes0, Bs0, Mcs0, Ucs0 = simulate_learning(0.9,.20, -0.27, 1, sim_length, 0, ParaCalib2(σ=0.0 ), ParaCalib2(σ = 0.0 ))


Js1, ns1, Js2, ns2 = fp(0,ParaCalib2())

Jsc, nsc, Jsc2, nsc2 = fp(0,ParaCalib2())



(figT2, ax2_bin) = subplots(1, 2, figsize = (20, 10))

ax2 = ax2_bin[1,1]

ax2.plot(Mc1, color = (0.24, 0.44, 0.54), linewidth = 3.0, linestyle = "-.")
ax2.plot(Mc2,color=(.541, .2, .141),linewidth=3.0,linestyle="--")
ax2.plot(Mc4, color = (0.0, 0.2, 0.5), linewidth = 3.0)
ax2.plot(Mc3,color=(0.0, .4, .0),linewidth=3.0,linestyle=":")
ax2.axis([0.00, 50, -3.30,0.0])
ax2.set_xlabel("horizon", fontsize=18)
ax2.set_ylabel(L"M_t", fontsize=18)
ax2.tick_params("both", labelsize = 12)
ax2.legend([L"γ=0.01",L"γ=0.17", L"γ^*=0.2", L"γ=0.5"], fontsize = 16, frameon = 0, loc = "lower right")


ax2 = ax2_bin[2,1]

ax2.plot(Uc1, color = (0.24, 0.44, 0.54), linewidth = 3.0, linestyle = "-.")
ax2.plot(Uc2,color=(.541, .2, .141),linewidth=3.0,linestyle="--")
ax2.plot(Uc4, color = (0.0, 0.2, 0.5), linewidth = 3.0)
ax2.plot(Uc3, color=(0.0, .4, .0),linewidth=3.0,linestyle=":")
ax2.set_xlabel("horizon", fontsize=18)
ax2.set_ylabel(L"U_t", fontsize=18)
ax2.tick_params("both", labelsize = 12)
ax2.legend([L"γ=0.01",L"γ=0.17", L"γ^*=0.2", L"γ=0.5"], fontsize = 16, frameon = 0, loc = "upper right")

#savefig("/Users/billbranch/Dropbox/Unemployment Stocks II/latest draft/figs/IRF_demand_gains.png")

# ax2 = ax2_bin[1,1]

# ax2.plot(Mc4, color = (0.0, 0.2, 0.5), linewidth = 2.0)
# ax2.plot(Mcs0,color=(.541, .2, .141),linewidth=2.0,linestyle="--")
# ax2.set_xlabel("horizon", fontsize=12)
# ax2.set_ylabel(L"M_t", fontsize=12)
# ax2.legend([L"σ=0.2",L"σ=0.0"], fontsize = 12, frameon = 0, loc = "upper right")

# ax2 = ax2_bin[1,2]

# ax2.plot(Uc4, color = (0.0, 0.2, 0.5), linewidth = 2.0)
# ax2.plot(Ucs0,color=(.541, .2, .141),linewidth=2.0,linestyle="--")
# ax2.set_xlabel("horizon", fontsize=12)
# ax2.set_ylabel(L"U_t", fontsize=12)
# ax2.legend([L"σ=0.2",L"σ=0.0"], fontsize = 12, frameon = 0, loc = "upper right")

# suptitle("Preference shock")

display(figT2)

close(figT2)

# (figDemand, ax2) = subplots(1,1, figsize = (10,10))
# ax2.plot(Uc4, color = (0.0, 0.2, 0.5), linewidth = 2.0)
# ax2.plot(Ucs0,color=(.541, .2, .141),linewidth=2.0,linestyle="--")
# ax2.set_xlabel("horizon", fontsize=12)
# ax2.set_ylabel(L"U_t", fontsize=12)
# ax2.legend([L"σ=0.2",L"σ=0.0"], fontsize = 12, frameon = 0, loc = "upper right")
# savefig("/Users/billbranch/Dropbox/Unemployment Stocks II/latest draft/figs/IRF_demand_U.png")
# close(figDemand)


nrms4 = .002466-rfunc(Jsim4[1], nsim4[1], ParaCalib2())
nrms0 = .002466-rfunc(Js0[1], ns0[1], ParaCalib2(σ=0.0))
sprd4 = [100.0*((.002466-rfunc(Jsim4[t], nsim4[t], ParaCalib2()))/nrms4-1.0) for t in 1:40]
sprd0 = [100.0*((.002466-rfunc(Js0[t],ns0[t], ParaCalib2(σ=0.0)))/nrms0-1.0) for t in 1:40]

(figsprd, ax2_bin) = subplots(3,1, figsize = (13,13))

axsp = ax2_bin[1,1]
axsp.plot(sprd4,color = (0.0, 0.2, 0.5), linewidth = 3.0)
axsp.plot(sprd0,color = (.541, .2, .141), linewidth = 3.0, linestyle = "--")
axsp.set_xlabel("horizon", fontsize=18)
axsp.set_ylabel("spread (%) ", fontsize=18)
axsp.tick_params("both", labelsize = 12)
axsp.legend([L"σ=0.2",L"σ=0.0"], fontsize = 16, frameon = 0, loc = "lower right")


axsp = ax2_bin[2,1]

axsp.plot(Mc4, color = (0.0, 0.2, 0.5), linewidth = 3.0)
axsp.plot(Mcs0,color=(.541, .2, .141),linewidth=3.0,linestyle="--")
axsp.set_xlabel("horizon", fontsize=18)
axsp.set_ylabel(L"M_t", fontsize=18)
axsp.tick_params("both", labelsize = 12)
axsp.legend([L"σ=0.2",L"σ=0.0"], fontsize = 16, frameon = 0, loc = "upper right")

axsp = ax2_bin[3,1]

axsp.plot(Uc4, color = (0.0, 0.2, 0.5), linewidth = 3.0)
axsp.plot(Ucs0,color=(.541, .2, .141),linewidth=3.0,linestyle="--")
axsp.set_xlabel("horizon", fontsize=18)
axsp.set_ylabel(L"U_t", fontsize=18)
axsp.tick_params("both", labelsize = 12)
axsp.legend([L"σ=0.2",L"σ=0.0"], fontsize = 16, frameon = 0, loc = "upper right")


display(figsprd)
#savefig("/Users/billbranch/Dropbox/Unemployment Stocks II/latest draft/figs/IRF_demand_all.png")
close(figsprd)



# expenditure shocks

sim_length = 48

Jsim12, nsim12, Msim12, Usim12, Jesim12, nesim12, Bsim12, Mc12, Uc12 = simulate_learning(0.9,.01, 0.200, 2, sim_length, 0, ParaCalib2(), ParaCalib2())
Jsim22, nsim22, Msim22, Usim22, Jesim22, nesim22, Bsim22, Mc22, Uc22 = simulate_learning(0.9,.17, 0.200, 2, sim_length, 0, ParaCalib2(), ParaCalib2())
Jsim32, nsim32, Msim32, Usim32, Jesim32, nesim32, Bsim32, Mc32, Uc32 = simulate_learning(0.9,.50, 0.200, 2, sim_length, 0, ParaCalib2(), ParaCalib2())
Jsim42, nsim42, Msim42, Usim42, Jesim42, nesim42, Bsim42, Mc42, Uc42 = simulate_learning(0.9,.20, 0.200, 2, sim_length, 0, ParaCalib2(), ParaCalib2())

Js02, ns02, Ms02, Us02, Jes02, nes02, Bs02, Mcs02, Ucs02 = simulate_learning(0.9,.20, 0.200, 2, sim_length, 0, ParaCalib2(σ=0.0 ), ParaCalib2(σ = 0.0 ))


Js1, ns1, Js2, ns2 = fp(0,ParaCalib2())

Jsc, nsc, Jsc2, nsc2 = fp(0,ParaCalib2())



(figT22, ax22_bin) = subplots(1, 2, figsize = (20, 10))

ax22 = ax22_bin[1,1]


ax22.plot(Mc12, color = (0.24, 0.44, 0.54), linewidth = 3.0, linestyle = "-.")
ax22.plot(Mc22,color=(.541, .2, .141),linewidth=3.0,linestyle="--")
ax22.plot(Mc32,color=(0.0, .4, .0),linewidth=3.0,linestyle=":")
ax22.plot(Mc42, color = (0.0, 0.2, 0.5), linewidth = 3.0)
ax22.set_xlabel("horizon", fontsize=18)
ax22.set_ylabel(L"M_t", fontsize=18)
ax22.tick_params("both", labelsize = 12)
ax22.legend([L"γ=0.01",L"γ=0.17", L"γ^*=0.2", L"γ=0.5"], fontsize = 16, frameon = 0, loc = "upper right")


ax22 = ax22_bin[2,1]
ax22.plot(Uc12, color = (0.24, 0.44, 0.54), linewidth = 3.0, linestyle = "-.")
ax22.plot(Uc22,color=(.541, .2, .141),linewidth=3.0,linestyle="--")
ax22.plot(Uc32,color=(0.0, .4, .0),linewidth=3.0,linestyle=":")
ax22.plot(Uc42, color = (0.0, 0.2, 0.5), linewidth = 3.0)
ax22.set_xlabel("horizon", fontsize=18)
ax22.set_ylabel(L"U_t", fontsize=18)
ax22.tick_params("both", labelsize = 12)
ax22.legend([L"γ=0.01",L"γ=0.17", L"γ^*=0.2", L"γ=0.5"], fontsize = 16, frameon = 0, loc = "lower right")

#savefig("/Users/billbranch/Dropbox/Unemployment Stocks II/latest draft/figs/IRF_alpha_gains.png")
# ax22 = ax22_bin[1,1]
# ax22.plot(Mc42, color = (0.0, 0.2, 0.5), linewidth = 2.0)
# ax22.plot(Mcs02,color=(.541, .2, .141),linewidth=2.0,linestyle="--")
# ax22.set_xlabel("horizon", fontsize=12)
# ax22.set_ylabel(L"M_t", fontsize=12)
# ax22.legend([L"σ=0.2",L"σ=0.0"], fontsize = 12, frameon = 0, loc = "upper right")

# ax22 = ax22_bin[1,2]
# ax22.plot(Uc42, color = (0.0, 0.2, 0.5), linewidth = 2.0)
# ax22.plot(Ucs02,color=(.541, .2, .141),linewidth=2.0,linestyle="--")
# ax22.set_xlabel("horizon", fontsize=12)
# ax22.set_ylabel(L"U_t", fontsize=12)
# ax22.legend([L"σ=0.2",L"σ=0.0"], fontsize = 12, frameon = 0, loc = "lower right")

#suptitle("Shock to expenditure risk")

display(figT22)

close(figT22)

nrms4 = .002466-rfunc(Jsim42[1], nsim42[1], ParaCalib2())
nrms0 = .002466-rfunc(Js02[1], ns02[1], ParaCalib2(σ=0.0))
sprd4 = [100.0*((.002466-rfunc(Jsim42[t], nsim42[t], ParaCalib2()))/nrms4-1.0) for t in 1:40]
sprd0 = [100.0*((.002466-rfunc(Js02[t],ns02[t], ParaCalib2(σ=0.0)))/nrms0-1.0) for t in 1:40]

(figsprd, axsp) = subplots(1,1, figsize = (10,10))
#ax22 = ax22_bin[1,2]
axsp.plot(sprd4,color = (0.0, 0.2, 0.5))
axsp.plot(sprd0,color = (.541, .2, .141), linestyle = "--")
axsp.set_xlabel("horizon", fontsize=12)
axsp.set_ylabel("spread (%) ", fontsize=12)
axsp.legend([L"σ=0.2",L"σ=0.0"], fontsize = 12, frameon = 0, loc = "lower left")

display(figsprd)
#savefig("/Users/billbranch/Dropbox/Unemployment Stocks II/latest draft/figs/IRF_alpha_sprd.png")
close(figsprd)

(figsprd, ax2_bin) = subplots(3,1, figsize = (13,13))

axsp = ax2_bin[1,1]
axsp.plot(sprd4,color = (0.0, 0.2, 0.5), linewidth = 3.0)
axsp.plot(sprd0,color = (.541, .2, .141), linewidth = 3.0, linestyle = "--")
axsp.set_xlabel("horizon", fontsize=18)
axsp.set_ylabel("spread (%) ", fontsize=18)
axsp.tick_params("both", labelsize = 12)
axsp.legend([L"σ=0.2",L"σ=0.0"], fontsize = 16, frameon = 0, loc = "center right")


axsp = ax2_bin[2,1]

axsp.plot(Mc42, color = (0.0, 0.2, 0.5), linewidth = 3.0)
axsp.plot(Mcs02,color=(.541, .2, .141),linewidth=3.0,linestyle="--")
axsp.set_xlabel("horizon", fontsize=18)
axsp.set_ylabel(L"M_t", fontsize=18)
axsp.tick_params("both", labelsize = 12)
axsp.legend([L"σ=0.2",L"σ=0.0"], fontsize = 16, frameon = 0, loc = "center right")

axsp = ax2_bin[3,1]

axsp.plot(Uc42, color = (0.0, 0.2, 0.5), linewidth = 3.0)
axsp.plot(Ucs02,color=(.541, .2, .141),linewidth=3.0,linestyle="--")
axsp.set_xlabel("horizon", fontsize=18)
axsp.set_ylabel(L"U_t", fontsize=18)
axsp.tick_params("both", labelsize = 12)
axsp.legend([L"σ=0.2",L"σ=0.0"], fontsize = 16, frameon = 0, loc = "center right")

display(figsprd)
#savefig("/Users/billbranch/Dropbox/Unemployment Stocks II/latest draft/figs/IRF_alpha_all.png")
close(figsprd)

# Ag shocks

sim_length = 148

Jsim13, nsim13, Msim13, Usim13, Jesim13, nesim13, Bsim13, Mc13, Uc13 = simulate_learning(0.9,.01, -.85, 3, sim_length, 0, ParaCalib2(), ParaCalib2())
Jsim23, nsim23, Msim23, Usim23, Jesim23, nesim23, Bsim23, Mc23, Uc23 = simulate_learning(0.9,.17, -0.85, 3, sim_length, 0, ParaCalib2(), ParaCalib2())
Jsim33, nsim33, Msim33, Usim33, Jesim33, nesim33, Bsim33, Mc33, Uc33 = simulate_learning(0.9,.2, -0.85, 3, sim_length, 0, ParaCalib2(), ParaCalib2())
Jsim43, nsim43, Msim43, Usim43, Jesim43, nesim43, Bsim43, Mc43, Uc43 = simulate_learning(0.9,.50, -0.85, 3, sim_length, 0, ParaCalib2(), ParaCalib2())

Js03, ns03, Ms03, Us03, Jes03, nes03, Bs03, Mcs03, Ucs03 = simulate_learning(0.9,.20, -0.85, 3, sim_length, 0, ParaCalib2(σ=0.0 ), ParaCalib2(σ = 0.0 ))



r4 = [rfunc(Jsim43[t], nsim43[t], ParaCalib2())/rfunc(Jsim43[1], nsim43[1], ParaCalib2()) for t in 1:40]
r0 = [rfunc(Js03[t], ns03[t], ParaCalib2())/rfunc(Js03[1], ns03[1], ParaCalib2()) for t in 1:40]

(figT23, ax23_bin) = subplots(1, 2, figsize = (20, 10))

ax23 = ax23_bin[1,1]

ax23.plot(Mc13, color = (0.24, 0.44, 0.54), linewidth = 3.0, linestyle = "-.")
ax23.plot(Mc23,color=(.541, .2, .141),linewidth=3.0,linestyle="--")
ax23.plot(Mc33,color=(0.0, 0.2, 0.5),linewidth=3.0)
ax23.plot(Mc43, color = (0.0, .4, .0),linestyle=":", linewidth = 3.0)
ax23.set_xlabel("horizon", fontsize=18)
ax23.set_ylabel(L"M_t", fontsize=18)
ax23.tick_params("both", labelsize = 12)
ax23.legend([L"γ=0.01",L"γ=0.17", L"γ^*=0.2", L"γ=0.5"], fontsize = 16, frameon = 0, loc = "upper right")


ax23 = ax23_bin[2,1]

ax23.plot(Uc13, color = (0.24, 0.44, 0.54), linewidth = 3.0, linestyle = "-.")
ax23.plot(Uc23,color=(.541, .2, .141),linewidth=3.0,linestyle="--")
ax23.plot(Uc33,color=(0.0, .2, .5),linewidth=3.0)
ax23.plot(Uc43, color = (0.0, .4, .0),linestyle=":", linewidth = 3.0)
ax23.set_xlabel("horizon", fontsize=18)
ax23.set_ylabel(L"U_t", fontsize=18)
ax23.tick_params("both", labelsize = 12)
ax23.legend([L"γ=0.01",L"γ=0.17", L"γ^*=0.2", L"γ=0.5"], fontsize = 16, frameon = 0, loc = "lower right")

display(figT23)
close(figT23)
#savefig("/Users/billbranch/Dropbox/Unemployment Stocks II/latest draft/figs/IRF_public_gains.png")


# ax23 = ax23_bin[1,1]


# ax23.plot(Mc43, color = (0.0, 0.2, 0.5), linewidth = 2.0)
# ax23.plot(Mcs03,color=(.541, .2, .141),linewidth=2.0,linestyle="--")
# ax23.set_xlabel("horizon", fontsize=12)
# ax23.set_ylabel(L"M_t", fontsize=12)
# ax23.legend([L"σ=0.2",L"σ=0.0"], fontsize = 12, frameon = 0, loc = "upper right")

# ax23 = ax23_bin[1,2]

# ax23.plot(Uc43, color = (0.0, 0.2, 0.5), linewidth = 2.0)
# ax23.plot(Ucs03,color=(.541, .2, .141),linewidth=2.0,linestyle="--")
# ax23.set_xlabel("horizon", fontsize=12)
# ax23.set_ylabel(L"U_t", fontsize=12)
# ax23.legend([L"σ=0.2",L"σ=0.0"], fontsize = 12, frameon = 0, loc = "lower right")

# suptitle("Shock to public liquidity")

display(figT23)


close(figT23)

(figsprd, ax23) = subplots(1,1, figsize = (10,10))
ax23.plot(Mc33, color = (0.0, 0.2, 0.5), linewidth = 2.0)
ax23.plot(Mcs03,color=(.541, .2, .141),linewidth=2.0,linestyle="--")
ax23.set_xlabel("horizon", fontsize=12)
ax23.set_ylabel(L"M_t", fontsize=12)
ax23.legend([L"σ=0.2",L"σ=0.0"], fontsize = 12, frameon = 0, loc = "upper right")

display(figsprd)
#savefig("/Users/billbranch/Dropbox/Unemployment Stocks II/latest draft/figs/IRF_public_M.png")
close(figsprd)

(figsprd, ax23) = subplots(1,1, figsize = (10,10))
ax23.plot(Uc33, color = (0.0, 0.2, 0.5), linewidth = 2.0)
ax23.plot(Ucs03,color=(.541, .2, .141),linewidth=2.0,linestyle="--")
ax23.set_xlabel("horizon", fontsize=12)
ax23.set_ylabel(L"U_t", fontsize=12)
ax23.legend([L"σ=0.2",L"σ=0.0"], fontsize = 12, frameon = 0, loc = "lower right")

display(figsprd)
#savefig("/Users/billbranch/Dropbox/Unemployment Stocks II/latest draft/figs/IRF_public_U.png")
close(figsprd)

nrms4 = .002466-rfunc(Jsim33[1], nsim33[1], ParaCalib2())
nrms0 = .002466-rfunc(Js03[1], ns03[1], ParaCalib2(σ=0.0))
sprd4 = [100.0*((.002466-rfunc(Jsim33[t], nsim33[t], ParaCalib2()))/nrms4-1.0) for t in 1:40]
sprd0 = [100.0*((.002466-rfunc(Js03[t],ns03[t], ParaCalib2(σ=0.0)))/nrms0-1.0) for t in 1:40]

(figsprd, axsp) = subplots(1,1, figsize = (10,10))
#ax22 = ax22_bin[1,2]
axsp.plot(sprd4,color = (0.0, 0.2, 0.5))
axsp.plot(sprd0,color = (.541, .2, .141), linestyle = "--")
axsp.set_xlabel("horizon", fontsize=12)
axsp.set_ylabel("spread (%) ", fontsize=12)
axsp.legend([L"σ=0.2",L"σ=0.0"], fontsize = 12, frameon = 0, loc = "lower right")

display(figsprd)
#savefig("/Users/billbranch/Dropbox/Unemployment Stocks II/latest draft/figs/IRF_public_sprd.png")
close(figsprd)

(figsprd, ax2_bin) = subplots(3,1, figsize = (13,13))

axsp = ax2_bin[1,1]
axsp.plot(sprd4,color = (0.0, 0.2, 0.5), linewidth = 3.0)
axsp.plot(sprd0,color = (.541, .2, .141), linewidth = 3.0, linestyle = "--")
axsp.set_xlabel("horizon", fontsize=18)
axsp.set_ylabel("spread (%) ", fontsize=18)
axsp.tick_params("both", labelsize = 12)
axsp.legend([L"σ=0.2",L"σ=0.0"], fontsize = 16, frameon = 0, loc = "center right")


axsp = ax2_bin[2,1]

axsp.plot(Mc33, color = (0.0, 0.2, 0.5), linewidth = 3.0)
axsp.plot(Mcs03,color=(.541, .2, .141),linewidth=3.0,linestyle="--")
axsp.set_xlabel("horizon", fontsize=18)
axsp.set_ylabel(L"M_t", fontsize=18)
axsp.tick_params("both", labelsize = 12)
axsp.legend([L"σ=0.2",L"σ=0.0"], fontsize = 16, frameon = 0, loc = "center right")

axsp = ax2_bin[3,1]

axsp.plot(Uc33, color = (0.0, 0.2, 0.5), linewidth = 3.0)
axsp.plot(Ucs03,color=(.541, .2, .141),linewidth=3.0,linestyle="--")
axsp.set_xlabel("horizon", fontsize=18)
axsp.set_ylabel(L"U_t", fontsize=18)
axsp.tick_params("both", labelsize = 12)
axsp.legend([L"σ=0.2",L"σ=0.0"], fontsize = 16, frameon = 0, loc = "center right")

display(figsprd)
#savefig("/Users/billbranch/Dropbox/Unemployment Stocks II/latest draft/figs/IRF_public_all.png")
close(figsprd)

#### Perfect storm experiment

function tempeq_ps(Je::Float64, ne::Float64, n::Float64, FixFlag::Int64, para::ParaCalib2)

    function obj(x::Float64)

       return Jfunc(nfunc(n, x, para), Je, ne, FixFlag,para)-x

    end

    return fzero(x -> obj(x), 0.5)

end

function simulate_learning_ps(shock_length::Int64, ρ::Float64, gn::Float64, shock::Float64, sim_length::Int64, FixFlag::Int64, para::ParaCalib2, para_n::ParaCalib2)

    # shock_length = # of periods that perfect storm shock lasts
    # ρ = decay to perfect storm

    Jsim = Array{Float64}(undef, sim_length)
    nsim = similar(Jsim)
    Msim = similar(Jsim)
    Usim = similar(Jsim)
    Je = similar(Jsim)
    ne = similar(Jsim)
    

    Js1, ns1, Js2, ns2 = fp(0, ParaCalib2())
    J1=Js1
    n1=ns1

    Je[1] = .75*J1
    ne[1] = n1

    

    Bt = Array{Float64}(undef, sim_length)
    αt = similar(Bt)
    λt = similar(Bt)
    kt = similar(Bt)
    w1t = similar(Bt)
    Agt = similar(Bt)
    ξt = similar(Bt)

    # Calibrated parameters, governs L.O.M. after shock_length

    Bs = 16.64
    αs = 0.00444
    λs = 0.8
    ks = 4.4016
    w1s = 0.97
    Ags = 4.528
    ξs = 0.5

    @unpack B, α, λ, k, w1, Ag = para
    @unpack ξ = para_n

    Bt[1] = B
    αt[1] = α
    λt[1] = λ
    kt[1] = k
    w1t[1] = w1
    Agt[1] = Ag
    ξt[1] = ξ

    

    Jsim[1] = J1
    nsim[1] = n1
    Msim[1] = Jsim[1]*nsim[1]
    Usim[1] = 1-n1
    

    for t in 2:sim_length

        Je[t] = Je[t-1] + gn * (Jsim[t-1] - Je[t-1])
        ne[t] = ne[t-1] + gn * (nsim[t-1] - ne[t-1])


        Bt[t] = Bt[t-1]-ρ*(Bt[t-1]-Bs)
        αt[t] = αt[t-1] - ρ*(αt[t-1]-αs)
        λt[t] = λt[t-1] - ρ*(λt[t-1]-λs)
        kt[t] = kt[t-1] - ρ*(kt[t-1]-ks)
        w1t[t] = w1t[t-1] - ρ*(w1t[t-1]-w1s)
        Agt[t] = Agt[t-1] - ρ*(Agt[t-1]-Ags)
        ξt[t] = ξt[t-1] - ρ*(ξt[t-1]-ξs)

        
        if t<= shock_length
            Jsim[t] = tempeq_ps(Je[t], ne[t], nsim[t-1], FixFlag, ParaCalib2(B=Bt[t], α=αt[t], λ=λt[t], k=kt[t], w1=w1t[t], Ag=Agt[t], ξ=ξt[t]))
            nsim[t] = nfunc(nsim[t-1], Jsim[t], ParaCalib2(B=Bt[t], α=αt[t], λ=λt[t], k=kt[t], w1=w1t[t], Ag=Agt[t], ξ=ξt[t]))
        else
            
            Jsim[t] = tempeq_ps(Je[t], ne[t], nsim[t-1], 0, ParaCalib2())
            nsim[t] = nfunc(nsim[t-1], Jsim[t], ParaCalib2())
        
        end
        Msim[t] = Jsim[t]*nsim[t]
        Usim[t] = 1-nsim[t]

    end
    

    #return Jsim, nsim, Msim, Usim, Je, ne, αt
    return Msim, Usim

end

sim_length = 100

Msimmulti1, Usimmulti1 = simulate_learning_ps(12,1.0,1.0, 0.0, sim_length, 1, ParaCalib2(α=0.04,λ=0.1, zbar=0.8, δ=.035, ξ=.45, k=1.0, w1=1.0, Ag=1.25,γ=1.0, B=18.0, σ=.2), ParaCalib2(α=0.04,λ=0.1, zbar=0.8, δ=.035, ξ=.45, k=1.0, w1=1.0, Ag=1.25,γ=1.0, B=18.0, σ=.2))
Msimmulti2, Usimmulti2 = simulate_learning_ps(24,1.0,1.0, 0.0, sim_length, 1, ParaCalib2(α=0.04,λ=0.1, zbar=0.8, δ=.035, ξ=.45, k=1.0, w1=1.0, Ag=1.25,γ=1.0, B=18.0, σ=.2), ParaCalib2(α=0.04,λ=0.1, zbar=0.8, δ=.035, ξ=.45, k=1.0, w1=1.0, Ag=1.25,γ=1.0, B=18.0, σ=.2))
Msimmulti3, Usimmulti3 = simulate_learning_ps(36,1.0,1.0, 0.0, sim_length, 1, ParaCalib2(α=0.04,λ=0.1, zbar=0.8, δ=.035, ξ=.45, k=1.0, w1=1.0, Ag=1.25,γ=1.0, B=18.0, σ=.2), ParaCalib2(α=0.04,λ=0.1, zbar=0.8, δ=.035, ξ=.45, k=1.0, w1=1.0, Ag=1.25,γ=1.0, B=18.0, σ=.2))



(figT3, ax3_bin) = subplots(2, 1, figsize = (10, 10))

ax3 = ax3_bin[1,1]
ax3.plot(Msimmulti1,color="black",linewidth=3.0,linestyle="--")
ax3.plot(Msimmulti2,color=(0.0, 0.2, 0.5), linewidth=3.0, linestyle=":")
ax3.plot(Msimmulti3,color=(0.0, 0.4, 0.0))
ax3.set_xlabel("horizon", fontsize=18)
ax3.set_ylabel(L"M_t", fontsize=18)
ax3.legend([L"T=12",L"T=24",L"T=36"], fontsize = 16, frameon = 0, loc = "lower right")
#ax3.plot(Msimmulti1)


ax3 = ax3_bin[2,1]
ax3.plot(Usimmulti1,color="black",linewidth=3.0,linestyle="--")
ax3.plot(Usimmulti2,color=(0.0, 0.2, 0.5), linewidth=3.0, linestyle=":")
ax3.plot(Usimmulti3,color=(0.0, 0.4, 0.0))
ax3.set_ylabel(L"U_t", fontsize = 18)


display(figT3)
#savefig("/Users/billbranch/Dropbox/Unemployment Stocks II/latest draft/figs/ps_sims.png")
close(figT3)


