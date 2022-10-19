using PlotlyJS
using PyPlot
# to construct Sobol sequences
using QuantEcon, Roots
using DataFrames
using Parameters
using PyCall
using Temporal, Dates
# standard library components
using Printf
using Statistics, JSON3, DelimitedFiles

#cd("C:\\Users\\BIZtech\\Dropbox\\Unemployment Stocks II\\JEDC revision\\Programs\\Julia 25 July 2022")
#cd("C:\\Users\\TJSEM\\Dropbox\\Unemployment Stocks II\\JEDC revision\\Programs\\Julia 25 July 2022")
include("functions_LUS.jl")
include("time_series_fun.jl")
include("business_cycle_solution_functions_08172022.jl")

mom_data = readdlm("mom_data.csv")
json_string = read("params_calib.json", String)
para = JSON3.read(json_string)
#δ, zbar, w0, w1, ϕ, ρ, α, B, σ, k, A, ξ, Bg, λ, η = para

function criterion(moms_model, moms_data)
    # SMM weighted sum of squared moment errors
    e = @. (moms_model - moms_data)/moms_data
    #e'*W*e
    return (e'*e)[1]
end

sim_length = 100_000
initial = Date(1800, 1, 1)
final = initial + Month(sim_length-1)
date_seq = initial:Month(1):final

# Running the combined
function smm_objective(params_est, mom_data, date_seq; coefs::Union{Matrix{Float64}, Nothing}=nothing)
    ρx, ρB, ρα, ρη, σx, σB, σα, ση = params_est
    global p = Params(ρx=ρx, σx=σx, ρB=ρB, σB=σB,
                ρα=ρα, σα=σα, ρη=ρη, ση=ση)
    m = Model(p, deg=2)
    global coefs = coefs
    coefs, solve_time = solve(m, tol=1e-7, damp=0.3, coefs_init=coefs)
    # simulate the model
    sim = Simulation(m, coefs, capT=100_000)
    " Log deviations from stationary mean "
    fields = [:y, :u, :M]
    out = reduce(hcat, [100 .*log.(getfield(sim, x)./mean(getfield(sim,x))) for x in [:y, :u, :M]])
   
    df = TS(out, date_seq);
    df.fields = fields

    # convert to quarterly
    df_q = collapse(df, eoq(df.index), fun=mean);
    # Apply bk filter
    cycle = mapcols(col -> bkfilter(col, wl=6, wu=200, K=12), DataFrames.DataFrame(df_q.values))
    #cycle = mapcols(col -> bkfilter(col, wl=6, wu=200, K=12), DataFrames.DataFrame(df_q.values,:auto))
    DataFrames.rename!(cycle, fields)

    #Extract moments
    mom_mod = moments(cycle, :y, [:u, :M], var_names=[:y, :u, :M])
    #pprint(mom_data)
    mom_mod = reduce(vcat,[mom_mod[!,:SD], mom_mod[[1,3],4], mom_mod[1, 5],
    mom_mod[!, 6], mom_mod[!, 7]])
    crit = criterion(mom_mod, mom_data)
    return crit
end


#ρx, ρB, ρα, ρη, σx, σB, σα, ση
lower = [0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001]
upper = [0.999, 0.999, 0.999, 0.999, 0.3, 0.3, 0.3, 0.3]
initial = [0.979, 0.979, 0.979, 0.979, 0.01, 0.01, 0.01, 0.01]
# Try 
smm_objective(initial, mom_data, date_seq)

inner_optimizer = GradientDescent()
#inner_optimizer = LBFGS()
results = optimize(params -> smm_objective(params, mom_data, date_seq), lower, upper,
    initial, Fminbox(inner_optimizer))
params_est = results.minimizer


#α_t = shock_transform.(p.α, exp.(sim.ξα))
#α_t = shock_transform(p.α, 1.0)








