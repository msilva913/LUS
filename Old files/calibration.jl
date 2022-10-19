
using PyPlot
using LaTeXStrings, KernelDensity
using Parameters, CSV, StatsBase, Statistics, Random
using NLsolve, Dierckx, Distributions, ArgParse
using LinearAlgebra, QuadGK, Roots, Optim
using DataFrames, CSV
using LeastSquaresOptim
using Polynomials
using JSON3, Printf
include("functions_LUS.jl")

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
                rho_annual=0.03, mu=1.2):
    " Targets used for calibration "
    # if os.path.isfile('calibration_data'):
    #    calibration_data = pickle.load(open('calibration_data', 'rb'))
    #    u_level, e_level, u_new, jf, s, w_hour, weekly_hours, spread, debt_gdp = calibration_data
    # else:
    u_disc = fred.get_series('U5RATE').loc[init:final]
    print(u_disc.mean())
    u_level = fred.get_series('UNEMPLOY').loc[init:final] #number unemployed in thousands
    e_level = fred.get_series('CE16OV').loc[init:final] #number employed in thousands
    u_new = fred.get_series('UEMPLT5').loc[init:final] #unemployed for less than 5 weeks
    jf = 1-(u_level[1:len(u_level)-1]-u_new)/u_level# job finding rate series
    s = u_new/(e_level*(1-(1/2)*jf)) #separation rate series
    
    # Krishnamurthy and Vissing-Jorgensen data
    path = r'C:\Users\TJSEM\Dropbox\Unemployment Stocks II\Python programs\KV_data'
    #path = r'C:\Users\BIZTech\Dropbox\Unemployment Stocks II\Python programs\KV_data'
    os.chdir(path)
    KV_data = pd.read_stata('jpe_data.dta', index_col='year')
    path = r'C:\Users\TJSEM\Dropbox\Unemployment Stocks II\Python programs'
    #path = r'C:\Users\BIZTech\Dropbox\Unemployment Stocks II\Python programs'
    os.chdir(path)
    spread = KV_data['aaatreas']
    debt_gdp = np.exp(KV_data['lndebtgdp'])
        
    stock_cap_gdp = fred.get_series('DDDM01USA156NWDB').resample('A').mean()
        #debt_stock = (debt_gdp/stock_cap_gdp).mean()
        #save_object([u_level, e_level, u_new, jf, s, w_hour, weekly_hours, spread, debt_gdp], 'calibration_data')
        
    
    #A 1% increase in debt to GDP reduces the interest rate spread by 0.746%.
    out = stock_cap_gdp, spread, rho_annual, zbar, jf, s, debt_gdp, mu, replace
    return out
"""

function calibrate(targets)
    # Targets
    # 1-3) Treasury demand 
    # 4) Market tightness
    # 5) Stock market cap to gdp
    # 6) Labor share of income

    stock_cap_GDP = targets[1].mean()/100
    spread = targets[2].values
    rho_annual = targets[3]
    zbar = targets[4]
    jf = targets[5].values
    s = targets[6].values
    debt_gdp = targets[7].values
    μ = targets[8]
    #rep = targets[9]
    rep = 0.6
    """
    a) Construct separation rate, job finding probability using unemployment
       #data, and tightness 
    """
    
    jf_mean = nanmean(jf)
    δ = nanmean(s)
    n = 1.0 - nanmean(s./(s +jf))
    q = 1.0 -(1.0-1/3)^4
    θ = jf_mean/q    
    ξ = 0.5
    A = jf_mean/(θ^(1-ξ))
    σ = μ - 1.0
    #ρ = (1+ rho_annual)^(1/12) - 1.0
    λ = 0.8
    #η = 0.7
  
    function Treasury_demand(param, xdata, ydata)
        # Extract parameters
        k, B, α, w0, ϕ, ρ, η = sqrt.(param.^2)

        # Monthly debt-to-gdp is scaled up by 12 (since GDP is divided by 12)
        xdata_mon = xdata*12
        spread_list = zeros(size(xdata_mon))
        J_list = similar(spread_list)
        n_list = similar(spread_list)
        Bg_list = similar(spread_list)
        theta_list = similar(spread_list)
        z_list = similar(spread_list)
        w_list = similar(spread_list)

        for (i, x) in enumerate(xdata_mon)

            para = ParaCalib2(δ=δ, zbar=zbar, w0=w0, ϕ=ϕ, ρ=ρ, α=α, B=B, σ=σ, k=k, A=A, ξ=ξ, λ=λ, η=η)
            # parameterization consistent with debt-to-GDP ratio
            para = ParaTrans(x, para, 0)
            # steady state
            ss = SteadyState(para, 0)
            spread_list[i] = ss.spread_bond*100*12
            J_list[i] = ss.J 
            n_list[i] = ss.n 
            Bg_list[i] = para.Bg
            theta_list[i] = ss.θ
            z_list[i] = ss.z 
            w_list[i] = ss.w
         end

        # convert spread to annualized percentage points
        out = zero(param)
        # Distance between model spread and data
        out[1:3] = (spread_list-ydata)./ydata
        # Distance from market tightness to target
        out[4] = (theta_list[2] - θ)/θ
        # Distance from outside option and revenue
        out[5] = (rep*z_list[2]-w0)/w0
        #Labor share
        #out[5] = (w_list[2]/z_list[2]-0.64)/0.64
        #out[5] = (ϕ*z_list[2]/w_list[2]-0.5)
        # Stock market cap to GDP
        out[6] = J_list[2]/(12*z_list[2]) - stock_cap_GDP    
        return out, Bg_list[2]
    end

    # Quadratic fit to K_V data
    poly_fit = Polynomials.fit(debt_gdp, spread, 2)
    # Target points for calibration
    xdata = [0.2, median(debt_gdp), 0.7]
    ydata = poly_fit.(xdata)

    # Loss function
    Treasury_demand_bas(param) = Treasury_demand(param, xdata, ydata)[1]

    " Least squares "
    params_init = [0.4, 19.043, 0.005, 0.705, 0.568]
    #LeastSquaresOptim.optimize(Treasury_demand_bas, params_init, Dogleg())

    py"""
    res = least_squares(fun=$Treasury_demand_bas, 
                    x0=np.array((9.21, 16.68, 0.00446, 0.62, 0.4, 0.00341, 0.4815)),
         verbose=2, max_nfev=150, x_scale='jac', method='trf',
        bounds = ((0, 0, 0, 0.6, 0.0, 0.02/12, 0), (np.inf, np.inf, 0.99, np.inf, 0.99, 0.006, 0.99) ),
        ftol=1e-9, xtol=1e-9)

    #save_object(res, "optimization_object_ext")
    """
    params_opt = py"res.x"
    k, B, α, w0, ϕ, ρ, η = params_opt
    out, Bg = Treasury_demand(params_opt, xdata, ydata)
    params = ParaCalib2(δ=δ, zbar=zbar, w0=w0, ϕ=ϕ, ρ=ρ, α=α, B=B, σ=σ, k=k, A=A, ξ=ξ, λ=λ, Bg=Bg, η=η)

    return params, poly_fit
end

function plot_Treasury_demand(params, debt_gdp, spread, poly_fit)
    xdata = range(0.2, stop = 0.8, length=6)|>collect 
    # Monthly debt-to-gdp is 12 times annual
    xdata_mon = xdata*12
    spread_list = similar(xdata_mon)

    for (i, x) in enumerate(xdata_mon)
        # Instantiate ParamsLUStrans
        para = ParaTrans(x, params, 0)
        ss = SteadyState(para, 0)
        spread_list[i] = ss.spread_bond*100*12
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
    #PyPlot.savefig("/Users/BIZtech/Dropbox/Unemployment Stocks II/latest draft/figs/Treasury_demand_curve.pdf")
    PyPlot.savefig("/Users/TJSEM/Dropbox/Unemployment Stocks II/latest draft/figs/Treasury_demand_curve.pdf")
    end

py"""def Tab(params):
        δ, zbar, w0, w1, ϕ, ρ, α, B, σ, k, A, ξ, Bg, λ, η = params
        params = [δ, zbar, w0, ϕ, ρ, α, B, σ, k, A, ξ, Bg, λ, η]
        names =['$\\delta$' , '$\overline{z}$', '$w_0$', '$\\phi$', '$\\rho$', '$\\alpha$',
                '$B$', '$\\sigma$', '$k$', '$A$', '$\\xi$', '$B^g$', '$\\lambda$', '$\\eta$']
        strat=['mean separation rate', 
            'normalization',
            'replacement ratio', 
            'Treasury demand',
            'risk free rate',
            'Treasury demand', 
            'Treasury demand', 
            'Price to average cost',
            'consistency with market tightness', 
            'Job finding rate',
            ' Elasticity of matching function with respect to unemployment ',
            'Treasury demand',
            'Fraction of HH with access to credit',
            'Treasury demand']
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

params, poly_fit = calibrate(targets)

spread = targets[2].values
debt_gdp = targets[7].values

plot_Treasury_demand(params, debt_gdp, spread, poly_fit)

@unpack δ, zbar, w0, w1, ϕ, ρ, α, B, σ, k, A, ξ, Bg, λ, η = params

para = ParaCalib2(w0=w0,ϕ=ϕ, ρ=ρ, α=α, B=B, σ=σ, k=k, A=A, Bg=Bg, λ=λ, η=η)
ss = SteadyState(para, 0)
@unpack n, J, spread_bond, spread_equity, r_a, r_b, w, z, M, w, z, θ = ss

@printf "\n Bond spread =  %.4f" spread_bond*12*100
@printf "\n Equity spread = %.4f" spread_equity*12*100
@printf "\n Annual rate on illiquid bond = %.4f" ρ*12*100
@printf "\n Labor share of income = %.2f" w/z
@printf "\n Stock market cap to GDP = %.2f" J/(12*z)
@printf "\n Debt to GDP = %.2f" Bg/(12*z)
@show 12*100(spread_bond-spread_equity)


# Save output
tab = [δ, zbar, w0, w1, ϕ, ρ, α, B, σ, k, A, ξ, Bg, λ, η]

open("params_calib.json", "w") do io
    JSON3.pretty(io, tab)
end
json_string = read("params_calib.json", String)
par = JSON3.read(json_string)
δ, zbar, w0, w1, ϕ, ρ, α, B, σ, k, A, ξ, Bg, λ, η = par


tab = py"Tab($tab)"
