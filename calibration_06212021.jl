
using PyPlot
using LaTeXStrings, KernelDensity
using Parameters, CSV, StatsBase, Statistics, Random
using NLsolve, Dierckx, Distributions, ArgParse
using LinearAlgebra, QuadGK, Roots, Optim
using DataFrames, CSV
using LeastSquaresOptim
using Polynomials
using JSON3
include("functions_LUS.jl")

# path = r'C:\Users\TJSEM\Dropbox\Unemployment Stocks II\Python programs'
# os.chdir(path)
# from model_class_02222021 import ParamsLUS, ParamsLUSTrans
using PyCall
#ENV["PYTHON"] = raw"C:\ProgramData\Anaconda3\python.exe"
ENV["PYTHON"] = raw"C:\\Users\\Administrator\\anaconda3\\python.exe"
using Conda
#Conda.pip_interop(true)
#Conda.pip("install", "fredapi")
#Conda.pip("install", "scipy")
Conda.pip("install", "astropy")
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
        
        #stock_cap_gdp = fred.get_series('DDDM01USA156NWDB').resample('A').mean()
        #debt_stock = (debt_gdp/stock_cap_gdp).mean()
        #save_object([u_level, e_level, u_new, jf, s, w_hour, weekly_hours, spread, debt_gdp], 'calibration_data')
        
    ##health share
    
    #A 1% increase in debt to GDP reduces the interest rate spread by 0.746%.
    out = replace, spread, rho_annual, zbar, jf, s, debt_gdp, mu
    return out
"""

function calibrate(targets)
    rep = targets[1]
    spread = targets[2].values
    rho_annual = targets[3]
    zbar = targets[4]
    jf = targets[5].values
    s = targets[6].values
    debt_gdp = targets[7].values
    μ = targets[8]
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
    ρ = (1+ rho_annual)^(1/12) - 1.0
    λ = 0.8
  
    function Treasury_demand(param, xdata, ydata)
        k, B, α, w0, ϕ = sqrt.(param.^2)

        # Monthly debt-to-gdp is scaled up by 12 (since GDP is divided by 12)
        xdata_mon = xdata*12
        spread_list = zeros(size(xdata_mon))
        J_list = similar(spread_list)
        n_list = similar(spread_list)
        Ag_list = similar(spread_list)
        theta_list = similar(spread_list)
        z_list = similar(spread_list)
        w_list = similar(spread_list)

        for (i, x) in enumerate(xdata_mon)

            para = ParaCalib2(δ=δ, zbar=zbar, w0=w0, ϕ=ϕ, ρ=ρ, α=α, B=B, σ=σ, k=k, A=A, ξ=ξ, λ=λ)
            # parameterization consistent with debt-to-GDP ratio
            para = ParaTrans(0, x, para)
            # steady state
            ss = steady_state(0, para)
            spread_list[i] = ss.spread*100*12
            J_list[i] = ss.J 
            n_list[i] = ss.n 
            Ag_list[i] = para.Ag 
            theta_list[i] = ss.θ
            z_list[i] = ss.z 
            w_list[i] = ss.w
         end

        # convert spread to annualized percentage points
        out = zeros(5)
        # Distance between model spread and data
        out[1:3] = (spread_list-ydata)./ydata
        # Distance from market tightness to target
        out[4] = (theta_list[2] - θ)/θ
        # Distance from outside option and revenue
        out[5] = (rep*z_list[2]-w0)/w0
        return out, Ag_list[2]
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
                    x0=np.array((4.41, 16.65, 0.004448, 0.7186, 0.04702)),
         verbose=2, max_nfev=150, x_scale='jac', method='dogbox',
        bounds = ((0, 0, 0, 0, 0), (np.inf, np.inf, 0.99, np.inf, 0.99) ),
        ftol=1e-8/3, xtol=1e-8/3)
    #save_object(res, "optimization_object_ext")
    """
    params_opt = py"res.x"
    k, B, α, w0, ϕ = params_opt
    out, Ag = Treasury_demand(params_opt, xdata, ydata)
    params = ParaCalib2(δ=δ, zbar=zbar, w0=w0, ϕ=ϕ, ρ=ρ, α=α, B=B, σ=σ, k=k, A=A, ξ=ξ, λ=λ, Ag=Ag)


    return params, poly_fit
end

function plot_Treasury_demand(params, debt_gdp, spread, poly_fit)
    xdata = range(0.2, stop = 0.8, length=6)|>collect 
    # Monthly debt-to-gdp is 12 times annual
    xdata_mon = xdata*12
    spread_list = similar(xdata_mon)

    for (i, x) in enumerate(xdata_mon)
        # Instantiate ParamsLUStrans
        para = ParaTrans(0, x, params)
        ss = steady_state(0, para)
        spread_list[i] = ss.spread*100*12
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
    end

function main()
    targets = py"gen_targets()"

    # open("targets.json", "w") do io
    #     JSON3.pretty(io, targets)
    # end

    params, poly_fit = calibrate(targets)

    spread = targets[2].values
    debt_gdp = targets[7].values

    plot_Treasury_demand(params, debt_gdp, spread, poly_fit)

    # Save output
    @unpack δ, zbar, w0, w1, ϕ, ρ, α, B, σ, k, A, ξ, Ag, λ, γ = params
    tab = [δ, zbar, w0, w1, ϕ, ρ, α, B, σ, k, A, ξ, Ag, λ, γ]

    open("params_calib.json", "w") do io
        JSON3.pretty(io, tab)
    end
end

main()
json_string = read("params_calib.json", String)
par = JSON3.read(json_string)
δ, zbar, w0, w1, ϕ, ρ, α, B, σ, k, A, ξ, Ag, λ, γ = par

py"""def Tab(params):
        δ, zbar, w0, w1, ϕ, ρ, α, B, σ, k, A, ξ, Ag, λ, γ = params
        params = [δ, zbar, w0, ϕ, ρ, α, B, σ, k, A, ξ, Ag, λ]
        names =['$\\delta$' , '$\overline{z}$', '$w_0$', '$\\phi$', '$\\rho$', '$\\alpha$',
                '$B$', '$\\sigma$', '$k$', '$A$', '$\\xi$', '$A^g$', '$\\lambda$']
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
            'Fraction of HH with access to credit']
        t = Table([names, tuple(params), strat], names=('Parameter','Values', 'Calibration Strategy')
        )
        t['Values'].format='0.4g'
        ascii.write(t, format='latex')  
        #ascii.write(t, format='commented_header')
        return t  
    """

tab = py"Tab($par)"
para = ParaCalib2()
ss = steady_state(0, para)
@unpack n, J, spread, w, z, M = ss
ss.spread*12*100
L = M+Ag
labor_share = w/z
private_share = M/(M+Ag)

