
using PyPlot
using Parameters, StatsBase, Statistics, Random
using NLsolve, Distributions, ArgParse
using LinearAlgebra, Roots, Optim
using DataFrames, CSV
using LeastSquaresOptim
using Polynomials
using JSON3, Printf


include("steady_state_convexity.jl")

# path = r'C:\Users\TJSEM\Dropbox\Unemployment Stocks II\Python programs'
# os.chdir(path)
using PyCall
#ENV["PYTHON"] = raw"C:\ProgramData\Anaconda3\python.exe"
ENV["PYTHON"] = raw"C:\\Users\\Administrator\\anaconda3\\python.exe"
using Conda
#Conda.pip_interop(true)
#Conda.pip("install", "fredapi")
#Conda.pip("install", "scipy")
#Conda.pip("install", "astropy")
nanmean(x) = mean(filter(!isnan,x))
nanmean(x,y) = mapslices(nanmean,x,y)

py"""
import os
import pdb
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from collections import namedtuple
from fredapi import Fred
fred = Fred(api_key = 'd35aabd7dc07cd94481af3d1e2f0ecf3	')
from astropy.table import Table
from astropy.io import ascii
import pickle
"""


py"""def gen_targets(init='1959', final='2019', replace=0.71, zbar=1,
                rho_annual=0.03, mu=1.2, hiring_cost_share=0.129, annual_dest_rate = 0.025,
                annual_dep_rate = 0.1):
    " Targets used for calibration "
    # if os.path.isfile('calibration_data'):
    #    calibration_data = pickle.load(open('calibration_data', 'rb'))
    #    u_level, e_level, u_new, jf, s, w_hour, weekly_hours, spread, debt_gdp = calibration_data
    # else:
    u_rate = fred.get_series('UNRATE').loc[init:final].mean()
    u_level = fred.get_series('UNEMPLOY').loc[init:final] #number unemployed in thousands
    e_level = fred.get_series('CE16OV').loc[init:final] #number employed in thousands
    u_new = fred.get_series('UEMPLT5').loc[init:final] #unemployed for less than 5 weeks
    jf = 1-(u_level[1:len(u_level)-1]-u_new)/u_level# job finding rate series
    s = u_new/(e_level*(1-(1/2)*jf)) #separation rate series
    
    # Krishnamurthy and Vissing-Jorgensen data
    #path = r'C:\Users\TJSEM\Dropbox\Unemployment Stocks II\Python programs\KV_data'
    path = r'C:\Users\BIZTech\Dropbox\Unemployment Stocks II\Python programs\KV_data'
    os.chdir(path)
    KV_data = pd.read_stata('jpe_data.dta', index_col='year')
    #path = r'C:\Users\TJSEM\Dropbox\Unemployment Stocks II\Python programs'
    path = r'C:\Users\BIZTech\Dropbox\Unemployment Stocks II\Python programs'
    os.chdir(path)
    spread = KV_data['aaatreas']
    debt_gdp = np.exp(KV_data['lndebtgdp'])
        
    stock_cap_gdp = fred.get_series('DDDM01USA156NWDB').resample('A').mean()
        #debt_stock = (debt_gdp/stock_cap_gdp).mean()
        #save_object([u_level, e_level, u_new, jf, s, w_hour, weekly_hours, spread, debt_gdp], 'calibration_data')
        
    
    #A 1% increase in debt to GDP reduces the interest rate spread by 0.746%.
    out = stock_cap_gdp, spread, rho_annual, zbar, jf, s, debt_gdp, mu, replace, hiring_cost_share, u_rate, annual_dep_rate
    return out
"""

function calibrate(targets)
    # Targets
    # 1-3) Treasury demand 
    # 4) Market tightness
    # 5) Stock market cap to gdp
    # 6) Capital stock to output
    # 7) Labor share of income
    # 8) Hiring cost share

    stock_cap_GDP = targets[1].mean()/100
    spread = targets[2].values
    rho_annual = targets[3]
    Z = targets[4]
    f = targets[5].values
    τ = targets[6].values
    debt_gdp = targets[7].values
    μ = targets[8]
    rep = targets[9]
    hiring_cost_share = targets[10]
    u = targets[11]
    annual_dep_rate = targets[12]

    annual_dest_rate = 0.03
    δ = 1-(1-annual_dest_rate)^(1/12)
    δ_k = 1-(1-annual_dep_rate)^(1/12)

    # Composite parameters    
    # Correct job finding and vacancy filling probablities
    f = nanmean(f)/(1-δ)
    q = (1.0 -(1.0-1/3)^4)/(1-δ)

    # Separations taken from Beveridge curve
    τ = nanmean(τ)
    #τ = 1- (1-δ)*(1-s)
    s = (τ-δ)/(1-δ)
    u = τ/(τ+f)
    # τ = u/(1-u)*f
    # s = (τ - δ)/(1-δ)

    """
    a) Construct separation rate, job finding probability using unemployment
       #data, and tightness 
    """
    
    θ = f/q    
    ξ = 0.5
    A = f/(θ^(1-ξ))
    #η = μ - 1.0
    η = 0.2
    #ρ = (1+ rho_annual)^(1/12) - 1.0
    λ = 0.0
    α = 0.3
    η_k = 0.0
    σ_1 = 1.1
  
    function Treasury_demand(param, xdata, ydata)
        # Extract parameters
        γ, ω, B, σ, σ_2, b, ϕ, ρ, η_a = sqrt.(param.^2)

        # Monthly debt-to-gdp is scaled up by 12 (since GDP is divided by 12)
        xdata_mon = xdata*12
        bond_spread_list = zeros(size(xdata_mon))
        M_list = similar(bond_spread_list)
        Y_list = similar(bond_spread_list)
        n_list = similar(bond_spread_list)
        Bg_list = similar(bond_spread_list)
        K_list = similar(bond_spread_list)
        θ_list = similar(bond_spread_list)
        z_list = similar(bond_spread_list)
        w_list = similar(bond_spread_list)
        κ_list = similar(bond_spread_list)
        r_a_list = similar(bond_spread_list)

        for (i, x) in enumerate(xdata_mon)

            para = ParaCalib(s=s, δ=δ, τ=τ, δ_k=δ_k, Z=Z, b=b, ϕ=ϕ, η=η, ρ=ρ, α=α, σ=σ, B=B, σ_1=σ_1, σ_2=σ_2, γ=γ, ω=ω,
                             A=A, ξ=ξ, Bg=0.0, η_a=η_a, η_k=0.0, λ=λ)

            # parameterization consistent with debt-to-GDP ratio
            para = ParaTrans(x, para)
            # steady state
            ss = steady_state(para)
            bond_spread_list[i] = ss.bond_spread
            M_list[i] = ss.M
            Y_list[i] = ss.Y
            n_list[i] = ss.n 
            Bg_list[i] = para.Bg
            θ_list[i] = ss.θ
            z_list[i] = ss.z 
            w_list[i] = ss.w
            κ_list[i] = ss.κ
            r_a_list[i] = ss.r_a
         end

        # convert spread to annualized percentage points
        out = zero(param)
        # Distance between model spread and data
        out[1:3] = 10*(bond_spread_list-ydata)./ydata

        # Distance from market tightness to target
        out[4] = (θ_list[2] - θ)/θ

        #  Consistency with job creation condition
        out[5] = (γ + κ_list[2])/q  - (1-δ)/(r_a_list[2]+τ)*(z_list[2]-w_list[2]-κ_list[2])

        # Ratio of hiring costs to wage: assoc. with γ
        out[6] = (γ-hiring_cost_share*q*w_list[2])/γ

        # Ratio of outside option to productivity (Hall and Milgrom 2008): assoc with b
        out[7] = (b-0.71*z_list[2])/b
        
        # Stock market cap to GDP: assoc. with ρ
        out[8] = M_list[2]/(12*Y_list[2]) - stock_cap_GDP    

        # Labor share of income: assoc. with ϕ
        out[9] = 100*(n_list[2]*w_list[2]/Y_list[2] - 0.64)/0.64

        return out, Bg_list[2]
    end

    # Quadratic fit to K_V data
    poly_fit = Polynomials.fit(debt_gdp, spread, 2)
    # Target points for calibration
    xdata = [0.2, median(debt_gdp), 0.7]
    ydata = poly_fit.(xdata)

    # Loss function
    Treasury_demand_bas(param) = Treasury_demand(param, xdata, ydata)[1]

    #γ, ω, B, σ, σ_2, b, ϕ, ρ, α, η_a, η_k
    # json_string = read("params_calib.json", String)
    # par = JSON3.read(json_string)
    # s, δ, δ_k, Z, b, ϕ, η, ρ, α, σ, B, _, σ_2, λ, γ, ω, A, ξ, Bg, η_a, η_k = par
    " Least squares "
    params_init = [0.26, 43.6, 12.33, 0.001, 0.85, 2.0, 0.68, 0.00288, 0.44]
    #params_init = [γ, ω, B, σ, σ_2, b, ϕ, ρ, η_a]
    res = LeastSquaresOptim.optimize(Treasury_demand_bas, params_init, LevenbergMarquardt(), store_trace=true, show_trace=true)
    #res = nlsolve(Treasury_demand_bas, params_init,  method = :trust_region, store_trace=true, show_trace=true)
   
    # py"""
    # res = least_squares(fun=$Treasury_demand_bas, 
    #                 x0=$res.minimizer,
    #      verbose=2, max_nfev=250, x_scale='jac', method='dogbox',
    #     bounds = ((0, 0, 0, 0, 0, 0, 0, 0.02/12, 0), (5.0, 200.0, 100.0, 0.6, 1.5, 5.0, 0.99, 0.1/12, 0.7) ),
    #     ftol=1e-10, xtol=1e-10)

    #save_object(res, "optimization_object_ext")
    #"""

    params_opt = py"res.x"
    γ, ω, B, σ, σ_2, b, ϕ, ρ, η_a = res.minimizer
    out, Bg = Treasury_demand(params_opt, xdata, ydata)
    para = ParaCalib(s=s, δ=δ, δ_k=δ_k, η=η, ρ=ρ, α=α, B=B, σ=σ, σ_1=σ_1, σ_2=σ_2, λ=λ, γ=γ, ω=ω, b=b, ϕ=ϕ, Bg=Bg, η_a=η_a, η_k=η_k)

    return para, poly_fit
end

function plot_Treasury_demand(params, debt_gdp, spread, poly_fit)
    xdata = range(0.2, stop = 0.8, length=6)|>collect 
    # Monthly debt-to-gdp is 12 times annual
    xdata_mon = xdata*12
    spread_list = similar(xdata_mon)

    for (i, x) in enumerate(xdata_mon)
        # Instantiate ParamsLUStrans
        para = ParaTrans(x, params)
        ss = steady_state(para)
        spread_list[i] = ss.bond_spread
        end

    fig, ax = subplots()
    ax.scatter(debt_gdp, spread, s=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 2])
    ax.set_xlabel("Debt/GDP")
    ax.set_ylabel("AAA- Treasury spread")
    ax.plot(xdata, poly_fit.(xdata), label="Fitted quadratic polynomial")
    ax.plot(xdata, spread_list,  label="Model Treasury demand")
    ax.legend()
    ax.set_title("Treasury demand curve")
    display(fig)
    PyPlot.savefig("/Users/BIZtech/Dropbox/Unemployment Stocks II/latest draft/figs/Treasury_demand_curve.pdf")
    #PyPlot.savefig("/Users/TJSEM/Dropbox/Unemployment Stocks II/latest draft/figs/Treasury_demand_curve.pdf")
    end

#@unpack s, δ, δ_k, Z, b, ϕ, η, ρ, α, σ, B, σ_1, σ_2, λ, γ, ω, A, ξ, Bg, η_a, η_k = para
py"""def Tab(params):
        s, δ, δ_k, Z, b, ϕ, η, ρ, α, σ, B, σ_1, σ_2, λ, γ, ω, A, ξ, Bg, η_a, η_k = params
        params = [s, δ, δ_k, Z, b, ϕ, η, ρ, α, σ, B, σ_1, σ_2, λ, γ, ω, A, ξ, Bg, η_a, η_k]
        names =['s', '$\\delta$' , '$\\delta_k$', '$Z$', '$b$', '$\\phi$', '$\\eta$', '$\\rho$', '$\\alpha$',
                 '$\\sigma$', '$B$', '$\\sigma_1$', '$\\sigma_2', '$\\lambda$', '$\\gamma$', '\\omega', 'A', 
                 '$\\xi$', '$B^g$', '\\eta_a', '\\eta_k']
        strat=['Mean separation rate', #s
                'Firm exit rate, Business Dynamics Statistics', #δ
                '10\%$ annual depreciation rate', #δ _k
                'Normalization', #Z
                'b=0.71z, Hall and Milgrom 2008', #b
            'Labor share of income=64\%$', #ϕ
            ' Fixed', #η
            'Stock market cap to output ratio=95\%$', #ρ
            'Capital to output ratio = 300\%$', #α
            ' Treasury demand', #σ
            'Treasury demand', #B 
            'Fixed', #σ_1
            'Treasury demand', #σ_2
            'Proportion of households with access to credit', #λ
            ' Hiring costs relative to wage (Silva and Toledo 2009)', #γ
            ' Market tightness', #ω
            'Job finding rate', #A
            'Petrongolo and Pissarides (1991)', #ξ
            'Median debt-to-output', #Bg
            'Treasury demand', #η_a
            'Treasury demand'] #η_k
        t = Table([names, tuple(params), strat], names=('Parameter','Values', 'Calibration Strategy')
        )
        t['Values'].format='0.4g'
        ascii.write(t, format='latex')  
        #ascii.write(t, format='commented_header')
        return t  
    """

targets = py"gen_targets()"

# open("targets.json", "w") do io
#     JSON3.pretty(io, targets)
# end

para, poly_fit = calibrate(targets)

spread = targets[2].values
debt_gdp = targets[7].values

plot_Treasury_demand(para, debt_gdp, spread, poly_fit)

@unpack s, δ, δ_k, Z, b, ϕ, η, ρ, α, σ, B, σ_1, σ_2, λ, γ, ω, A, ξ, Bg, η_a, η_k = para

para = ParaCalib(s=s, δ=δ, δ_k=δ_k, Z=Z, b=b, ϕ=ϕ, η=η, ρ=ρ, α=α, σ=σ, B=B, σ_1=σ_1, σ_2=σ_2, λ=λ, γ=γ,
    ω=ω, A=A, ξ=ξ, Bg=Bg, η_a=η_a, η_k=η_k)
ss = steady_state(para)
@unpack n, M, Y, bond_spread, equity_premium, r_a, z, M, w, z, θ, q, K = ss

@printf "\n Bond spread =  %.3f" bond_spread
@printf "\n Equity premium = %.3f" equity_premium
@printf "\n Annual rate on illiquid bond = %.3f" ρ*12*100
@printf "\n Labor share of income = %.2f" n*w/(Y)
@printf "\n Replacement ratio = %.2f" b/z
@printf "\n Hiring cost share = %.3f" γ/(q*w)
@printf "\n Stock market cap to GDP = %.2f" M/(12*Y)
@printf "\n Capital share of output = %.2f" K/(12*Y)
@printf "\n Debt to GDP = %.2f" Bg/(12*Y)
#@show 12*100(spread_bond-spread_equity)


# Save output
tab = [s, δ, δ_k, Z, b, ϕ, η, ρ, α, σ, B, σ_1, σ_2, λ, γ, ω, A, ξ, Bg, η_a, η_k]
#cd("C:\\Users\\TJSEM\\Dropbox\\Unemployment Stocks II\\JEDC revision\\Programs\\Julia 25 July 2022")
cd("C:\\Users\\BIZtech\\Dropbox\\Unemployment Stocks II\\JEDC revision\\Programs\\Julia 25 July 2022")
open("params_calib.json", "w") do io
    JSON3.pretty(io, tab)
end
json_string = read("params_calib.json", String)
par = JSON3.read(json_string)
s, δ, δ_k, Z, b, ϕ, η, ρ, α, σ, B, σ_1, σ_2, λ, γ, ω, A, ξ, Bg, η_a, η_k = par


table = py"Tab($tab)"
# py"""
# dat = Tab($tab)
# ascii.write(dat, Writer=ascii.Latex)
# """

