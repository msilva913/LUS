
using PyPlot
using LaTeXStrings, KernelDensity
using Parameters, CSV, StatsBase, Statistics, Random
using DataFrames
using ShiftedArrays
using QuantEcon
using MappedArrays
using TexTables
using TypedTables

function moments(dat, var; lags =2, verbose=true)
    #sd = DataFrames.colwise(std, dat)
    sd = map(std, eachcol(dat))
    RSD = sd./std(dat[!, var])

    corrs = [cor(dat[:, i], dat[:, var]) for i in 1:ncol(dat)]
    ac = zeros(ncol(dat), lags)

    for k in 2:(lags+1)
        ac[:, k-1] = [autocor(dat[:, i])[k] for i in 1:ncol(dat)]
    end

    mom = [names(dat) sd RSD corrs ac]
    mom = DataFrames.DataFrame(mom, :auto)
    #mom = convert(DataFrames.DataFrame, mom)
    rename!(mom, ["Variable", "SD", "RSD", "corrs", "Cor(x, x_{-1})", "Cor(x, x_{-2})"])
    Table(mom)
    if verbose
        mom[!, 2:end] = round.(mom[!, 2:end], sigdigits=3)
    end
    mom[!, :Variable] = [:SR, :Y, :L, :C, :TI]
    return mom
end

function moments(dat, var_RSD, var_corr; lags =2, 
    var_names=[:SR, :Y, :L, :C, :TI], verbose=true)
    #sd = DataFrames.colwise(std, dat)
    sd = map(std, eachcol(dat))
    RSD = sd./std(dat[!, var_RSD])

    corrs = zeros(ncol(dat), length(var_corr))
    lab = []
    for (i, var) in enumerate(var_corr)
        corrs[:, i] = [cor(dat[:, i], dat[:, var]) for i in 1:ncol(dat)]
        lab
    end
    cor_lab = ["Cor(x, $(var_corr[n]))" for n = 1:length(var_corr)]
    cor_lab = permutedims(cor_lab)
    
  
    ac = zeros(ncol(dat), lags)

    for k in 2:(lags+1)
        ac[:, k-1] = [autocor(of_eltype(Float64, dat[:, i]))[k] for i in 1:ncol(dat)]
    end

    mom = [names(dat) sd RSD corrs ac]
    #mom = DataFrames.DataFrame(mom, :auto)
    mom = DataFrames.DataFrame(mom)
    name = ["Variable" "SD" "RSD" cor_lab "Cor(x, x_{-1})" "Cor(x, x_{-2})"]
    DataFrames.rename!(mom, vec(name))
    #Table(mom)
    if verbose
        mom[!, 2:end] = round.(mom[!, 2:end], sigdigits=3)
    end
    mom[!, :Variable] = var_names
    return mom
end


function hamilton_filter(x; h=8)
    ones_col = ones(size(x))
    x_h = ShiftedArrays.lag(x, h)
    x_h1 = ShiftedArrays.lag(x_h, 1)
    x_h2 = ShiftedArrays.lag(x_h, 2)
    x_h3 = ShiftedArrays.lag(x_h, 3)
    X = [ones_col x x_h x_h1 x_h2 x_h3]
    X = DataFrame(X)
    # rename
    names!(X, [:ones_col, :x, :x_h, :x_h1, :x_h2, :x_h3])
    ols = lm(@formula(x ~ ones_col + x_h + x_h1 + x_h2 + x_h3), X)
    return residuals(ols)
end


function growth_filter(x)
    X = Pandas.DataFrame(x)
    X = diff(X, 1)
    X = X - mean(X)
    return X
end

# function linear_filter(x)
#     X = DataFrame()
#     X[!, :x] = x
#     T = range(1, size(X)[1], step=1)
#     X[!, :ones] = ones(size(x))
#     X[!, :time] = T
#     ols = lm(@formula(x~  time), X)
#     return residuals(ols)
# end  

function bk_filter(y::DataFrames.DataFrame; wl=6, wu=32, K=12)
    ### Arguments
    #y: data to be filtered
    #wu: upper cutoff frequencies
    #wl: lower cutoff frequencies
    #K: number of leads and lags of moving average

    # Returns
    #y_cycle: cyclical component
    #y_trend: trend component
    T = length(y)
    #T = nrow(y)
    w1 = 2pi/wu
    w2 = 2pi/wl
    b = vcat((w2-w1)/pi, [(sin(w2*i)-sin(w1*i))/(i*pi) for i=1:K])
    theta = (b[1] + 2*sum(b[2:end]))/(2K+1)
    B = b .- theta
    #vals = y.value
    #y[!, :cycle] = Vector{Union{Missing, Float64}}(undef, T)
    cycle = Vector{Union{Missing, Float64}}(undef, T)
    for t = K+1:T-K
        y[t,:cycle] = vals[1]*B[1] + (vals[t-1:-1:t-K]'*B[2:end]) + (vals[t+1:t+K]'*B[2:end])
    end
    y[!,:trend] = y[:, :value] - y[:, :cycle]
    return dropmissing(y)
end

function bkfilter(y::Vector{<:Union{Missing, Float64}}; wl=6, wu=32, K=12)
    ### Arguments
    #y: data to be filtered
    #wu: upper cutoff frequencies
    #wl: lower cutoff frequencies
    #K: number of leads and lags of moving average

    # Returns
    #y_cycle: cyclical component
    #y_trend: trend component
    T = length(y)
    w1 = 2pi/wu
    w2 = 2pi/wl
    b = vcat((w2-w1)/pi, [(sin(w2*i)-sin(w1*i))/(i*pi) for i=1:K])
    theta = (b[1] + 2*sum(b[2:end]))/(2K+1)
    B = b .- theta
    cycle = Vector{Union{Missing, Float64}}(undef, T)
    for t = K+1:T-K
        cycle[t] = y[1]*B[1] + (y[t-1:-1:t-K]'*B[2:end]) + (y[t+1:t+K]'*B[2:end])
    end
    #trend = y - cycle
    return filter!(!ismissing, cycle)
end

function bkfilter(y; wl=6, wu=32, K=12)
    ### Arguments
    #y: data to be filtered
    #wu: upper cutoff frequencies
    #wl: lower cutoff frequencies
    #K: number of leads and lags of moving average

    # Returns
    #y_cycle: cyclical component
    #y_trend: trend component
    T = length(y)
    w1 = 2pi/wu
    w2 = 2pi/wl
    b = vcat((w2-w1)/pi, [(sin(w2*i)-sin(w1*i))/(i*pi) for i=1:K])
    theta = (b[1] + 2*sum(b[2:end]))/(2K+1)
    B = b .- theta
    cycle = Vector{Union{Missing, Float64}}(undef, T)
    for t = K+1:T-K
        cycle[t] = y[1]*B[1] + (y[t-1:-1:t-K]'*B[2:end]) + (y[t+1:t+K]'*B[2:end])
    end
    #trend = y - cycle
    return filter!(!ismissing, cycle)
end

