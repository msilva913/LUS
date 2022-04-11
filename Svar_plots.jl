using CSV
using DataFrames
using Plots
using Colors, LaTeXStrings



df1 = DataFrame(CSV.File("/Users/billbranch/Dropbox/Unemployment Stocks II/Python programs/J_shock_irf.csv"))
df2 = DataFrame(CSV.File("/Users/billbranch/Dropbox/Unemployment Stocks II/Python programs/J_shock_irf_low_2.csv"))
df3 = DataFrame(CSV.File("/Users/billbranch/Dropbox/Unemployment Stocks II/Python programs/J_shock_irf_hi_2.csv"))



x = collect(range(0,99, length = 100))
p1 = plot(x,-df2[:,1], fillrange = -df3[:,1], fillalpha = 0.10, xlim = (0.0, 101.0), c=:gray, linecolor = :white, ylabel = L"\textrm{spread (b.p.)} ")
plot!(x, -df1[:,1], c = RGB(0.0, .200, .500), linewidth = 3)
p2 = plot(x,-df2[:,2], fillrange = -df3[:,2], fillalpha = 0.10, xlim = (0.0, 101.0), c=:gray, linecolor = :white, ylabel = L"\textrm{Stock mkt. cap.  } (\% )")
plot!(x,-df1[:,2],c = RGB(0.0, .200, .500), linewidth = 3)
p3 = plot(x,-100.0*df2[:,3]/7.2, fillrange = -100.0*df3[:,3]/7.2, fillalpha = 0.10, xlim = (0.0, 101.0), c=:gray, linecolor = :white, ylabel = L"\textrm{Unemp.  } (\% )")
plot!(x,-100.0*df1[:,3]/7.2, c = RGB(0.0, .20, .50), linewidth = 3)
#plot!(size = (800, 800), legend = false)
plot!(xlabel = L"T")
plot(p1, p2, p3, layout = (3,1), size = (800,800), legend = false)
savefig("/Users/billbranch/Dropbox/Unemployment Stocks II/latest draft/figs/IRF_svar.png")