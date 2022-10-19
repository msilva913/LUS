
using PyPlot
using LaTeXStrings
using Parameters, CSV, ArgParse
#using LinearAlgebra, Roots, Optim, NLsolve
using Printf
using DataFrames
using PrettyPrinting
using PlotlyJS
using MAT
using TexTables
using PyFormattedStrings
using DelimitedFiles
using QuantEcon
using FredData
using Pandas

function bk_filter(y; wl=6, wu=32, K=12)
    ### Arguments
    #y: data to be filtered
    #wu: upper cutoff frequencies
    #wl: lower cutoff frequencies
    #K: number of leads and lags of moving average

    # Returns
    #y_cycle: cyclical component
    #y_trend: trend component
    #T = length(y)
    T = nrow(y)
    w1 = 2pi/wu
    w2 = 2pi/wl
    b = vcat((w2-w1)/pi, [(sin(w2*i)-sin(w1*i))/(i*pi) for i=1:K])
    theta = (b[1] + 2*sum(b[2:end]))/(2K+1)
    B = b .- theta
    vals = y.value
    y[!, :cycle] = Vector{Union{Missing, Float64}}(undef, T)
    for t = K+1:T-K
        y[t,:cycle] = vals[1]*B[1] + (vals[t-1:-1:t-K]'*B[2:end]) + (vals[t+1:t+K]'*B[2:end])
    end
    y[!,:trend] = y[:, :value] - y[:, :cycle]
    return dropmissing(y)
end

f = Fred("d35aabd7dc07cd94481af3d1e2f0ecf3")
y = get_data(f, "GDPC1").data
y.value = log.(y.value)
select!(y, [:date,:value])
out = bk_filter(y)
cycle, trend = out.cycle, out.trend

fig, ax = subplots()
ax.plot(trend, label="trend", linewidth=2, alpha=0.6)
#ax.plot(y.value, label="raw", linewidth=2, alpha=0.6)
ax.legend()
display(fig)
std(cycle)

fig, ax = subplots()
ax.plot(cycle, label="cycle", linewidth=2, alpha=0.6)
ax.legend()
display(fig)


homedir()
#cd("C:\\Users\\BIZtech\\Dropbox\\Documents - Copy\\Research\\LUS heterogeneous agents\\Code\\Output")

# Read

