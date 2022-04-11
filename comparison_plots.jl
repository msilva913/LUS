
using PyPlot
using LaTeXStrings, KernelDensity
using Parameters, CSV, StatsBase, Statistics, Random
using NLsolve, Dierckx, Distributions, ArgParse
using LinearAlgebra, QuadGK, Roots, Optim

include("functions_LUS.jl")
# Calibrated parameters;


steps = 40
x_low = .00001
x_high = 0.05
xt = range(x_low, stop = x_high, length=steps)|>collect 
Mvec = zeros(steps, 3) # no. steps by model type
nvec = similar(Mvec)
Jvec = similar(Mvec)
svec = similar(Mvec)
zvec = similar(Mvec)
wvec = similar(Mvec)


para = ParaCalib2()
@unpack ρ = para
for i in 1:steps
    # Baseline
    ssT = fp(0, ParaCalib2(α = xt[i]))
    # Bewley Aiyagari
    ssT2 = fp(0, ParaCalib2(σ = 0.0, α = xt[i]))
    # interest rate channel: fix r = rho
    ssT3 = fp(0, ParaCalib2(α = xt[i]), ρ)
    
    # model types
    par_set = (ParaCalib2(α = xt[i]), ParaCalib2(σ = 0.0, α = xt[i]), ParaCalib2(α = xt[i]))
    ss_set = (ssT, ssT2, ssT3)
    for (j, (par, ss)) in enumerate(zip(par_set, ss_set))

        Jvec[i, j] = ss[1]
        nvec[i, j] = ss[2]
        Mvec[i, j] = ss[1]*ss[2]
        svec[i, j] = (ρ-rfunc(Jvec[i, j], nvec[i, j], par))*12*100
        zvec[i, j] = zfunc(Jvec[i, j], nvec[i, j], par)
        wvec[i, j] = wfunc(Jvec[i, j], nvec[i, j], Jvec[i, j], nvec[i, j], 0, par)
    end
    
end

uvec = 1.0 .- nvec

# List of results for each type
out = [[Mvec[:, j] uvec[:, j] svec[:, j] zvec[:, j] wvec[:, j]] for j in 1:3]




rcParams = PyPlot.matplotlib.rcParams
rcParams["font.size"] = 12
rcParams["axes.labelsize"] = 12
rcParams["axes.legendfontsize"] = 12
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8


# ylab = [ L"M", L"U", L"spread", L"z", L"w"]
# lab = [L"\sigma = 0.2", L"\sigma=0", L"\sigma = 0.2, J=J_{\rho}"]
# colors = [(0.0, 0.2, 0.5), (0.531, 0.2, 0.141), (0.3, 0.2, 0.5) ]
# linestyle = ["-", "--", ":"]
# fig, ax = subplots(nrows=3, ncols=2, figsize=(12, 14))
# for i in 1:3
#     for m in 1:3
#         for n in 1:2
#             # obtain correct index of variable
#             j = 2*(m-1) + (n-1) + 1
#             j = min(j, 5)
#             ax[m, n].plot(xt, out[i][:, j], label=lab[i], linewidth=2.5,
#             color=colors[i], linestyle=linestyle[i], alpha=0.8) #specification i variable j
#             ax[m, n].set(xlabel=L"\alpha", ylabel=ylab[j])
#             ax[m, n].legend(loc="best")
#         end
#     end
# end
# fig.delaxes(ax[3, 2])
# #savefig("/Users/billbranch/Dropbox/Unemployment Stocks II/latest draft/figs/IRF_alpha_gains.png")
# display(fig)       
# PyPlot.savefig("/Users/BIZTech/Dropbox/Unemployment Stocks II/latest draft/figs/channel_decomposition.pdf")

ylab = [ L"M", L"U", L"spread"]
lab = [L"\sigma = 0.2", L"\sigma=0", L"\sigma = 0.2, J=J_{\rho}"]
colors = [(0.0, 0.2, 0.5), (0.531, 0.2, 0.141), (0.3, 0.2, 0.5) ]
linestyle = ["-", "--", ":"]
fig, ax = subplots(nrows=1, ncols=3, figsize=(16, 5))
for i in 1:3
    for m in 1:3
        # obtain correct index of variable
        ax[m].plot(xt, out[i][:, m], label=lab[i], linewidth=2.5,
        color=colors[i], linestyle=linestyle[i], alpha=0.8) #specification i variable j
        ax[m].set(xlabel=L"\alpha", ylabel=ylab[m])
        ax[m].legend(loc="best")
        
    end
end
tight_layout()
display(fig)
PyPlot.savefig("/Users/BIZTech/Dropbox/Unemployment Stocks II/latest draft/figs/channel_decomposition.pdf")
