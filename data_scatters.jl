using TimeSeries, Plots, FredApi
    set_api_key("1eb4018cc8ead6ac6445763becf7bf42")
    gr(fmt=:png);
    cap = get_symbols("DDDM01USA156NWDB") #stock market capitalization
    capbuff = get_symbols("NCBEILQ027S","1996-01-01", "2017-01-01") # W. Buffet measure of cap.
    capbuffa = collapse(capbuff,year,first)  #annual
    ut = get_symbols("UNRATE","1996-01-01", "2017-01-01") 
    rt = get_symbols("AAA10Y","1996-01-01","2017-2-01")
    uta = collapse(ut,year,first)
    rta = collapse(rt,year,first)
    xvec = [Base.values(cap) Base.values(uta) Base.values(rta)]
    rsort = xvec[sortperm(xvec[:,3]),:]
    usort = xvec[sortperm(xvec[:,2]),:]
    csort = xvec[sortperm(xvec[:,1]),:]

    
    p1 = plot(rsort[1:11,1],rsort[1:11,2],  seriestype = :scatter, smooth = true, linewidth = 3, legend = false, title ="cap/urate sorted by spread")
    p1 = plot!(p1,rsort[12:22,1],rsort[12:22,2], linewidth = 3, seriestype = :scatter, smooth = true)
    p2 = plot(usort[1:11,1],usort[1:11,3],  seriestype = :scatter, smooth = true, linewidth = 3, legend = false, title ="cap/spread sorted by urate")
    p2 = plot!(p2,usort[12:22,1],usort[12:22,3], linewidth = 3, seriestype = :scatter, smooth = true)
    p3 = plot(csort[1:11,2],csort[1:11,3],  seriestype = :scatter, smooth = true, linewidth = 3, legend = false, title ="urate/spread sorted by cap")
    p3 = plot!(p3,csort[12:22,2],csort[12:22,3], linewidth = 3, seriestype = :scatter, smooth = true)
    
    
    plot(p1,p2,p3, layout = @layout [a b c])
    
    plot!(size=(1500,750))

    savefig("./Dropbox/Temp/datascatters.png")
