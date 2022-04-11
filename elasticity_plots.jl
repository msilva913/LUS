using FiniteDifferences
using Plots
using LaTeXStrings, KernelDensity
using Parameters, CSV, StatsBase, Statistics, Random
#using NLsolve,  Distributions, ArgParse
using NLsolve, Dierckx, Distributions, ArgParse
using LinearAlgebra, QuadGK, Roots, Optim

include("functions_LUS.jl")

function zfunc(J::Float64, n::Float64, para::ParaCalib2)

    @unpack zbar, σ, α, λ, B, Ag = para

    L = min(B, n*J+Ag)
    return zbar + (σ/(1+σ))*(α/n)*(λ*B+(1-λ)*L)

end

function yc(para::ParaCalib2)

    @unpack α, λ, B, Ag, σ = para

    ss = fp(0,para)
    M = ss[1]*ss[2]
    n = ss[2]

    return ((α/n)*(λ*B + (1-λ)* (M+Ag)))^(1/(1+σ))

end


function zf(para::ParaCalib2)

    @unpack zbar, σ = para

    return zbar + (σ/(1+σ))*(yc(para))^(1+σ)

end


function θf(s::Float64, para::ParaCalib2)
    " Market tightness in steady state"
    @unpack A, k, ρ, ξ = para

    ss = fp(0,para)
    J = ss[1]

    return (A*J/(k*(1+ρ-s)))^(1/ξ)
end

function qf(s::Float64, para::ParaCalib2)
    " Vacancy filling rate in steady state "
    @unpack A, ξ = para

    return A*θf(s,para)^(-ξ)

end

function wf(s::Float64, para::ParaCalib2)
    "Wage function in steady state "
    @unpack ϕ, w0, ρ, k, δ = para

    β = 1/(1+ρ)

    return (1-ϕ)*w0+ϕ*(zf(para)+β*(1+ρ-s)*k*θf(s,para))+ϕ*(1-δ)*k*(1-β*(1+ρ-s))/qf(s,para)

end

function Jf(s::Float64, para::ParaCalib2)
    " J function in steady state "
    @unpack ρ, δ = para

    return ((1+ρ-s)/(ρ+δ-s))*(zf(para)-wf(s,para))

end

function spreadElasticity(para::ParaCalib2)


    @unpack ρ, δ = para

    ss = fp(0, para)
    s_ss = ρ - rfunc(ss[1], ss[2], para)

    function obj(s::Float64)

        return Jf(s, para)

    end

    d = central_fdm(5,1)

    ds1 = d(obj, s_ss)
    return ds1*s_ss

end

function elasticityLabor(para::ParaCalib2)

    @unpack δ, zbar, w0, w1, ϕ, ρ, α, B, σ, k, A, ξ, Ag, λ = para

    ss = fp(0,para)
    M = ss[1]*ss[2]
    J = ss[1]
    n = ss[2]
    s = ρ - rfunc(ss[1], ss[2], para)
    z = zf(para)
    w = wf(s, para)
    θ = θf(s,para)
    q = A*θ^(-ξ)
    β = 1/(1+ρ)



    # ds/dm
    ds1 = B*α*(λ-1)*(1+ρ)
    ds2 = Ag + M + Ag*α*(λ-1)+(B-M)*α*(1-λ)
    ds = ds1/ds2^2

    # dz/dm
    dz1a = (α/n)*(-(M+Ag)*(λ-1)+B*λ)
    dz1 = σ*(λ-1)*(dz1a^(1/(1+σ)))^(1+σ)
    dz2 = (1+σ)*(Ag*(λ-1)+M*(λ-1)-B*λ)
    dz = dz1/dz2

    # dθ/ds
    dθ1 = (A*J/(k-k*s+k*ρ))^(1/ξ)
    dθ2 = ξ*(1-s+ρ)
    dθ = dθ1/dθ2

    # dθ/dJ

    dθJ = dθ1/(J*ξ)

    brack = (1+ρ-s)/(ρ+δ-s)
    dbrackds =  (1-δ)/(δ+ρ-s)^2

    ◬ = 1+brack*ϕ*β*(1+ρ-s)*k*dθJ+brack*q^(-2)*ϕ*(1-δ)*k*(1-β*(1+ρ-s))*A*ξ*θ^(-ξ-1)*dθJ

    

    dJ1 = dbrackds*ds*(z-w)
    dJ2 = brack*(dz-ϕ*dz+ϕ*β*k*θ*ds)
    dJ3 = -brack*(ϕ*β*(1+ρ-s)*k*dθ*ds+(1/q)*ϕ*(1-δ)*k*β*ds+q^(-2)*ϕ*(1-δ)*k*(1-β*(1+ρ-s))*A*ξ*θ^(-ξ-1)*ds)
    dJ = (1/◬)*(dJ1+dJ2+dJ3)

    n11 = A^(1/ξ)*k*δ*(ξ-1)*(s-ρ-1)*(J/(k-k*s+k*ρ))^(1/ξ)
    n12 = ξ*(J*δ-A^(1/ξ)*k*(s-ρ-1)*(J/(k-k*s+k*ρ))^(1/ξ))
    n1 = n11/n12^2

    n21 = -A^(1/ξ)*J*k*δ*(ξ-1)*(J/(k-k*s+k*ρ))^(1/ξ)
    n22 = ξ*n12^2
    n2 = n21/n22

    return n1*dJ + n2*ds

end

function elasticityLabor2(para::ParaCalib2)

    @unpack δ, zbar, w0, w1, ϕ, ρ, α, B, σ, k, A, ξ, Ag, λ = para

    ss = fp(0,para)
    M = ss[1]*ss[2]
    J = ss[1]
    n = ss[2]
    s = ρ - rfunc(ss[1], ss[2], para)
    z = zf(para)
    w = wf(s, para)
    θ = θf(s,para)
    q = A*θ^(-ξ)
    β = 1/(1+ρ)



    # ds/dm
    ds1 = B*α*(λ-1)*(1+ρ)
    ds2 = Ag + M + Ag*α*(λ-1)+(B-M)*α*(1-λ)
    ds = ds1/ds2^2

    ◬ = 1-((1-ξ)/ξ)*A^(1/ξ)*n^(1/ξ)*(M/(k*(1+ρ-s)))^((1-ξ)/ξ)

    dn1 = ((1-ξ)/ξ)*A^(1/ξ)*n^((1-ξ)/ξ)*(M/(k*(1+ρ-s)))^(1/ξ)*k*(1+ρ-s+(M*ds/(k*(1+ρ-s))^2))

    return n*dn1/◬

end

function elasticityEffects(para::ParaCalib2)

    @unpack δ, zbar, w0, w1, ϕ, ρ, α, B, σ, k, A, ξ, Ag, λ = para

    ss = fp(0,para)
    M = ss[1]*ss[2]
    J = ss[1]
    n = ss[2]
    s = ρ - rfunc(ss[1], ss[2], para)
    z = zf(para)
    w = wf(s, para)
    θ = θf(s,para)
    q = A*θ^(-ξ)
    β = 1/(1+ρ)



    # ds/dm
    ds1 = B*α*(λ-1)*(1+ρ)
    ds2 = Ag + M + Ag*α*(λ-1)+(B-M)*α*(1-λ)
    ds = ds1/ds2^2

    # dz/dm
    dz1a = (α/n)*(-(M+Ag)*(λ-1)+B*λ)
    dz1 = σ*(λ-1)*(dz1a^(1/(1+σ)))^(1+σ)
    dz2 = (1+σ)*(Ag*(λ-1)+M*(λ-1)-B*λ)
    dz = dz1/dz2

    # dθ/ds
    dθ1 = (A*J/(k-k*s+k*ρ))^(1/ξ)
    dθ2 = ξ*(1-s+ρ)
    dθ = dθ1/dθ2

    # dθ/dJ

    dθJ = dθ1/(J*ξ)

    brack = (1+ρ-s)/(ρ+δ-s)
    dbrackds =  (1-δ)/(δ+ρ-s)^2

    ◬ = 1+brack*ϕ*β*(1+ρ-s)*k*dθJ+brack*q^(-2)*ϕ*(1-δ)*k*(1-β*(1+ρ-s))*A*ξ*θ^(-ξ-1)*dθJ

    

    dJ1 = dbrackds*ds*(z-w)
    dJ2 = brack*(dz-ϕ*dz+ϕ*β*k*θ*ds)
    dJ3 = -brack*(ϕ*β*(1+ρ-s)*k*dθ*ds+(1/q)*ϕ*(1-δ)*k*β*ds+q^(-2)*ϕ*(1-δ)*k*(1-β*(1+ρ-s))*A*ξ*θ^(-ξ-1)*ds)
    dJ = (1/◬)*(dJ1+dJ2+dJ3)

    inteffect = dJ1*M
    demeffect = (dJ2+dJ3)*M

    return inteffect, demeffect

end

steps = 40
x_low = .00001
x_high = 0.05
xt = range(x_low, stop = x_high, length=steps)|>collect 
xt2 = range(x_low, stop = 1.0, length=steps)|>collect 
xt3 = range(x_low, stop = 10.0, length=steps)|>collect 
xt4 = range(x_low, stop = 0.5, length=steps)|>collect 
elas = Array{Float64}(undef,steps)
elas2 = similar(elas)
elas3 = similar(elas)
elas4 = similar(elas)

for i in 1:steps

    elas[i] = spreadElasticity(ParaCalib2(α = xt[i]))
    elas2[i] = spreadElasticity(ParaCalib2(λ = xt2[i]))
    elas3[i] = spreadElasticity(ParaCalib2(Ag = xt3[i]))
    elas4[i] = spreadElasticity(ParaCalib2(ϕ = xt4[i]))

end

# x = [0.05, 0.05 , NaN, 0.0, 0.05]
# y = [0.2, 0.1,  NaN, 1.5, 0.8]
x = [0.0044, 0.0044]
y = [0.5, 0.25]
#GR.setarrowsize(1)

ticker = 10
axer = 16

p1=plot(xt,elas, c = c = RGB(0.0, .200, .500), linewidth = 3)
scatter!([0.0044],[0.09962429325], c = RGB(.541, .2, .141), markersize = 10)
plot!(size = (800, 800), legend = false)
plot!(xtickfontsize=ticker,ytickfontsize=ticker, yguidefontsize = axer, xguidefontsize = axer, legendfontsize = 12)
plot!(grid = false)
plot!(xlabel = L"\alpha")
plot!(ylabel = L"\epsilon_{J,\textrm{spread}}")
annotate!(0.0055, 0.60, text("Calibration", RGB(.541, .2, .141), :center, 12))
plot!(x,y,  c = RGB(.541, .2, .141), arrow=true, arrowsize=0.5)

p2=plot(xt2,elas2, c = c = RGB(0.0, .200, .500), linewidth = 3)
scatter!([0.8],[0.09962429325], c = RGB(.541, .2, .141), markersize = 10)
plot!(size = (800, 800), legend = false)
plot!(xtickfontsize=ticker,ytickfontsize=ticker, yguidefontsize = axer, xguidefontsize = axer, legendfontsize = 12)
plot!(grid = false)
plot!(xlabel = L"\lambda")
plot!(ylabel = L"\epsilon_{J,\textrm{spread}}")
annotate!(0.8, 0.46, text("Calibration", RGB(.541, .2, .141), :center, 12))
x = [0.8, 0.8]
y = [0.4, 0.15]
plot!(x,y,  c = RGB(.541, .2, .141), arrow=true, arrowsize=0.5)

p3=plot(xt3,elas3, c = c = RGB(0.0, .200, .500), linewidth = 3)
x = [4.528, 4.528]
y = [0.22, 0.15]
scatter!([x[1]],[0.09962429325], c = RGB(.541, .2, .141), markersize = 10)
plot!(size = (800, 800), legend = false)
plot!(xtickfontsize=ticker,ytickfontsize=ticker, yguidefontsize = axer, xguidefontsize = axer, legendfontsize = 12)
plot!(grid = false)
plot!(xlabel = L"A_g")
plot!(ylabel = L"\epsilon_{J,\textrm{spread}}")
annotate!(4.528, 0.25, text("Calibration", RGB(.541, .2, .141), :center, 12))
plot!(x, y,  c = RGB(.541, .2, .141), arrow=true, arrowsize=0.5)

p4=plot(xt4,elas4, c = c = RGB(0.0, .200, .500), linewidth = 3)
x = [0.047354, 0.047354]
y = [0.16, 0.12]
scatter!([x[1]],[0.09962429325], c = RGB(.541, .2, .141), markersize = 10)
plot!(size = (800, 800), legend = false)
plot!(xtickfontsize=ticker,ytickfontsize=ticker, yguidefontsize = axer, xguidefontsize = axer, legendfontsize = 12)
plot!(grid = false)
plot!(xlabel = L"\phi")
plot!(ylabel = L"\epsilon_{J,\textrm{spread}}")
annotate!(x[1]+.05, 0.18, text("Calibration", RGB(.541, .2, .141), :center, 12))
plot!(x, y,  c = RGB(.541, .2, .141), arrow=true, arrowsize=0.5)

fig1 = plot(p1,p2,p3, p4, layout = (4,1))




steps = 40
x_low = .00001
x_high = 0.05
xt = range(x_low, stop = x_high, length=steps)|>collect 
xt2 = range(x_low, stop = 1.0, length=steps)|>collect 
xt3 = range(x_low, stop = 7.5, length=steps)|>collect 
xt4 = range(x_low, stop = 0.2, length=steps)|>collect 
elas = Array{Float64}(undef,steps)
elas2 = similar(elas)
elas3 = similar(elas)
elas4 = similar(elas)


for i in 1:steps

    elas[i] = elasticityLabor2(ParaCalib2(α = xt[i]))
    elas2[i] = elasticityLabor2(ParaCalib2(λ = xt2[i]))
    elas3[i] = elasticityLabor2(ParaCalib2(Ag = xt3[i]))
    elas4[i] = elasticityLabor2(ParaCalib2(ϕ = xt4[i]))

end

# x = [0.05, 0.05 , NaN, 0.0, 0.05]
# y = [0.2, 0.1,  NaN, 1.5, 0.8]
x = [0.0044, 0.0044]
y = [4, 2.75]
#GR.setarrowsize(1)

ticker = 10
axer = 16

p1b=plot(xt,elas, c = c = RGB(0.0, .200, .500), linewidth = 3)
scatter!([0.0044],[2.358195854929785], c = RGB(.541, .2, .141), markersize = 10)
plot!(size = (800, 800), legend = false)
plot!(xtickfontsize=ticker,ytickfontsize=ticker, yguidefontsize = axer, xguidefontsize = axer, legendfontsize = 12)
plot!(grid = false)
plot!(xlabel = L"\alpha")
plot!(ylabel = L"\epsilon_{n,M}")
annotate!(0.0055, y[1]+0.5, text("Calibration", RGB(.541, .2, .141), :center, 12))
plot!(x,y,  c = RGB(.541, .2, .141), arrow=true, arrowsize=0.5)

p2b=plot(xt2,elas2, c = c = RGB(0.0, .200, .500), linewidth = 3)
scatter!([0.8],[2.358195854929785], c = RGB(.541, .2, .141), markersize = 10)
plot!(size = (800, 800), legend = false)
plot!(xtickfontsize=ticker,ytickfontsize=ticker, yguidefontsize = axer, xguidefontsize = axer, legendfontsize = 12)
plot!(grid = false)
plot!(xlabel = L"\lambda")
plot!(ylabel = L"\epsilon_{n,M}")
x = [0.8, 0.8]
y = [2.47, 2.38]
annotate!(0.8, y[1]+0.02, text("Calibration", RGB(.541, .2, .141), :center, 12))
plot!(x,y,  c = RGB(.541, .2, .141), arrow=true, arrowsize=0.5)

p3b=plot(xt3,elas3, c = c = RGB(0.0, .200, .500), linewidth = 3)
x = [4.528, 4.528]
y = [2.46, 2.38]
scatter!([x[1]],[2.358195854929785], c = RGB(.541, .2, .141), markersize = 10)
plot!(size = (800, 800), legend = false)
plot!(xtickfontsize=ticker,ytickfontsize=ticker, yguidefontsize = axer, xguidefontsize = axer, legendfontsize = 12)
plot!(grid = false)
plot!(xlabel = L"A_g")
plot!(ylabel = L"\epsilon_{n,M}")
annotate!(4.528, y[1]+0.02, text("Calibration", RGB(.541, .2, .141), :center, 12))
plot!(x, y,  c = RGB(.541, .2, .141), arrow=true, arrowsize=0.5)

p4b=plot(xt4,elas4, c = c = RGB(0.0, .200, .500), linewidth = 3)
x = [0.047354, 0.047354]
y = [7.5, 3.75]
scatter!([x[1]],[2.358195854929785], c = RGB(.541, .2, .141), markersize = 10)
plot!(size = (800, 800), legend = false)
plot!(xtickfontsize=ticker,ytickfontsize=ticker, yguidefontsize = axer, xguidefontsize = axer, legendfontsize = 12)
plot!(grid = false)
plot!(xlabel = L"\phi")
plot!(ylabel = L"\epsilon_{n,M}")
annotate!(x[1], y[1]+1.0, text("Calibration", RGB(.541, .2, .141), :center, 12))
plot!(x, y,  c = RGB(.541, .2, .141), arrow=true, arrowsize=0.5)

fig2 = plot(p1b,p2b,p3b, p4b, layout = (4,1))

fig3 = plot(p1,p1b, p2, p2b, p3, p3b, p4, p4b, layout = (4,2), size = (1000,1000))
display(fig3)
#savefig("/Users/billbranch/Dropbox/Unemployment Stocks II/latest draft/figs/elasticities.png")

steps = 40
x_low = .00001
x_high = 0.05
xt = range(x_low, stop = x_high, length=steps)|>collect 
xt2 = range(x_low, stop = 1.0, length=steps)|>collect 
xt3 = range(x_low, stop = 7.5, length=steps)|>collect 
xt4 = range(x_low, stop = 0.2, length=steps)|>collect 
elas = Array{Float64}(undef,steps)
elas2 = similar(elas)
elas3 = similar(elas)
elas4 = similar(elas)
elasb = similar(elas)
elas2b = similar(elas)
elas3b = similar(elas)
elas4b = similar(elas)

for i in 1:steps

    elas[i], elasb[i] = elasticityEffects(ParaCalib2(α = xt[i]))
    elas2[i], elas2b[i] = elasticityEffects(ParaCalib2(λ = xt2[i]))
    elas3[i], elas3b[i] = elasticityEffects(ParaCalib2(Ag = xt3[i]))
    elas4[i], elas4b[i] = elasticityEffects(ParaCalib2(ϕ = xt4[i]))

end

# x = [0.05, 0.05 , NaN, 0.0, 0.05]
# y = [0.2, 0.1,  NaN, 1.5, 0.8]
x = [0.0044, 0.0044]
y = [-1.0,- 0.25]
#GR.setarrowsize(1)

axer = 18

p1c=plot(xt,elas, c = c = RGB(0.0, .200, .500), linewidth = 3)
scatter!([0.0044],[-0.13410913425432697], c = RGB(.541, .2, .141), markersize = 10)
plot!(size = (800, 800), legend = false)
plot!(xtickfontsize=ticker,ytickfontsize=ticker, yguidefontsize = axer, xguidefontsize = axer, legendfontsize = 12)
plot!(grid = false)
plot!(xlabel = L"\alpha")
plot!(ylabel = L"\epsilon_{\textrm{r-effect},M}")
annotate!(0.0044, y[1]-0.1, text("Calibration", RGB(.541, .2, .141), :center, 12))
plot!(x,y,  c = RGB(.541, .2, .141), arrow=true, arrowsize=0.5)

p2c=plot(xt2,elas2, c = c = RGB(0.0, .200, .500), linewidth = 3)
scatter!([0.8],[-0.13410913425432697], c = RGB(.541, .2, .141), markersize = 10)
plot!(size = (800, 800), legend = false)
plot!(xtickfontsize=ticker,ytickfontsize=ticker, yguidefontsize = axer, xguidefontsize = axer, legendfontsize = 12)
plot!(grid = false)
plot!(xlabel = L"\lambda")
plot!(ylabel = L"\epsilon_{\textrm{r-effect},M}")
x = [0.8, 0.8]
y = [-.4, -0.2]
annotate!(0.8, y[1]-0.10, text("Calibration", RGB(.541, .2, .141), :center, 12))
plot!(x,y,  c = RGB(.541, .02, .141), arrow=true, arrowsize=0.5)

p3c=plot(xt3,elas3, c = c = RGB(0.0, .200, .500), linewidth = 3)
x = [4.528, 4.528]
y = [-.4, -0.2]
scatter!([x[1]],[-0.13410913425432697], c = RGB(.541, .2, .141), markersize = 10)
plot!(size = (800, 800), legend = false)
plot!(xtickfontsize=ticker,ytickfontsize=ticker, yguidefontsize = axer, xguidefontsize = axer, legendfontsize = 12)
plot!(grid = false)
plot!(xlabel = L"A_g")
plot!(ylabel = L"\epsilon_{\textrm{r-effect},M}")
annotate!(4.528, y[1]-0.021, text("Calibration", RGB(.541, .2, .141), :center, 12))
plot!(x, y,  c = RGB(.541, .2, .141), arrow=true, arrowsize=0.5)

p4c=plot(xt4,elas4, c = c = RGB(0.0, .200, .500), linewidth = 3)
x = [0.047354, 0.047354]
y = [.3, -.05]
scatter!([x[1]],[-0.13410913425432697], c = RGB(.541, .2, .141), markersize = 10)
plot!(size = (800, 800), legend = false)
plot!(xtickfontsize=ticker,ytickfontsize=ticker, yguidefontsize = axer, xguidefontsize = axer, legendfontsize = 12)
plot!(grid = false)
plot!(xlabel = L"\phi")
plot!(ylabel = L"\epsilon_{\textrm{r-effect},M}")
annotate!(x[1], y[1]+0.1, text("Calibration", RGB(.541, .2, .141), :center, 12))
plot!(x, y,  c = RGB(.541, .2, .141), arrow=true, arrowsize=0.5)

x = [0.0044, 0.0044]
y = [0.2, 0.055]

p1d=plot(xt,elasb, c = c = RGB(0.0, .200, .500), linewidth = 3)
scatter!([0.0044],[0.0329852707286817], c = RGB(.541, .2, .141), markersize = 10)
plot!(size = (800, 800), legend = false)
plot!(xtickfontsize=ticker,ytickfontsize=ticker, yguidefontsize = axer, xguidefontsize = axer, legendfontsize = 12)
plot!(grid = false)
plot!(xlabel = L"\alpha")
plot!(ylabel = L"\epsilon_{\textrm{ad-effect},M}")
annotate!(0.0044, y[1]+0.02, text("Calibration", RGB(.541, .2, .141), :center, 12))
plot!(x,y,  c = RGB(.541, .2, .141), arrow=true, arrowsize=0.5)

p2d=plot(xt2,elas2b, c = c = RGB(0.0, .200, .500), linewidth = 3)
scatter!([0.8],[0.0329852707286817], c = RGB(.541, .2, .141), markersize = 10)
plot!(size = (800, 800), legend = false)
plot!(xtickfontsize=ticker,ytickfontsize=ticker, yguidefontsize = axer, xguidefontsize = axer, legendfontsize = 12)
plot!(grid = false)
plot!(xlabel = L"\lambda")
plot!(ylabel = L"\epsilon_{\textrm{ad-effect},M}")
x = [0.8, 0.8]
y = [.15, .04]
annotate!(0.8, y[1]+0.021, text("Calibration", RGB(.541, .2, .141), :center, 12))
plot!(x,y,  c = RGB(.541, .02, .141), arrow=true, arrowsize=0.5)

p3d=plot(xt3,elas3b, c = c = RGB(0.0, .200, .500), linewidth = 3)
x = [4.528, 4.528]
y = [.05, .036]
scatter!([x[1]],[0.0329852707286817], c = RGB(.541, .2, .141), markersize = 10)
plot!(size = (800, 800), legend = false)
plot!(xtickfontsize=ticker,ytickfontsize=ticker, yguidefontsize = axer, xguidefontsize = axer, legendfontsize = 12)
plot!(grid = false)
plot!(xlabel = L"A_g")
plot!(ylabel = L"\epsilon_{\textrm{ad-effect},M}")
annotate!(4.528, y[1]+0.0025, text("Calibration", RGB(.541, .2, .141), :center, 12))
plot!(x, y,  c = RGB(.541, .01, .141), arrow=true, arrowsize=0.5)

p4d=plot(xt4,elas4b, c = c = RGB(0.0, .200, .500), linewidth = 3)
x = [0.047354, 0.047354]
y = [7.5, 3.75]
scatter!([x[1]],[0.0329852707286817], c = RGB(.541, .2, .141), markersize = 10)
plot!(size = (800, 800), legend = false)
plot!(xtickfontsize=ticker,ytickfontsize=ticker, yguidefontsize = axer, xguidefontsize = axer, legendfontsize = 12)
plot!(grid = false)
plot!(xlabel = L"\phi")
plot!(ylabel = L"\epsilon_{\textrm{ad-effect},M}")
annotate!(x[1], y[1]+1.0, text("Calibration", RGB(.541, .2, .141), :center, 12))
plot!(x, y,  c = RGB(.541, .2, .141), arrow=true, arrowsize=0.5)



fig4 = plot(p1c, p1d, p2c, p2d, p3c, p3d,  layout = (3,2), size = (1100,1100))

display(fig1)

display(fig2)



display(fig4)
#savefig("/Users/billbranch/Dropbox/Unemployment Stocks II/latest draft/figs/effects.png")

