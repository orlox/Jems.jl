using DataFrames
using ForwardDiff, DataInterpolations
import ForwardDiff.Dual
using Jems.StellarModels
using Jems.DualExtrapolation; const de = DualExtrapolation
using Interpolations, Dierckx
using HDF5
using Jems.Constants
using DataInterpolations: CubicSpline
using CairoMakie, LaTeXStrings, MathTeXEngine, Makie.Colors, PlotUtils
basic_theme = Theme(fonts=(regular=texfont(:text), bold=texfont(:bold),
                           italic=texfont(:italic), bold_italic=texfont(:bolditalic)),
                    fontsize=30, size=(1000, 750), linewidth=7,
                    Axis=(xlabelsize=40, ylabelsize=40, titlesize=40, xgridvisible=false, ygridvisible=false,
                          spinewidth=2.5, xminorticksvisible=true, yminorticksvisible=true, xtickalign=1, ytickalign=1,
                          xminortickalign=1, yminortickalign=1, xticksize=14, xtickwidth=2.5, yticksize=14,
                          ytickwidth=2.5, xminorticksize=7, xminortickwidth=2.5, yminorticksize=7, yminortickwidth=2.5,
                          xticklabelsize=35, yticklabelsize=35, xticksmirrored=true, yticksmirrored=true),
                    Legend=(patchsize=(70, 10), framevisible=false, patchlabelgap=20, rowgap=10))
set_theme!(basic_theme)
##
gridpath = "DualRuns/DualGrid"
gridpath = "DualRuns/DualGrid2"
path = "DualRuns/DualGrid/logM_-0.1_X_0.7381_.history.hdf5"
get_logM(path) = parse(Float64, split(split(path, "logM_")[2], "_")[1])

#get ALL filepaths in gridpath
historypaths = filter(x -> occursin(".history.hdf5", x), readdir(gridpath))
profilepaths = filter(x -> occursin(".profiles.hdf5", x), readdir(gridpath))
N = length(historypaths)
println("$N history files found")
models = Dict()
modeltracks = Dict()
X_dual         = ForwardDiff.Dual{}(0.7381,  0.0,1.0,0.0)
Z_dual         = ForwardDiff.Dual{}(0.0134,  0.0,0.0,1.0)
Dfraction_dual = ForwardDiff.Dual{}(0.000312,0.0,0.0,0.0)
R_dual         = ForwardDiff.Dual{}(100*RSUN,0.0,0.0,0.0)

inititial_params_names = [:logM, :X, :Z, :Dfraction, :R]
all_logMs = []
i=8
only_use_these_logMs = [0.0,0.02]
only_use_these_logMs = []
for i in 1:N
    historypath = joinpath(gridpath, historypaths[i])
    profilepath = joinpath(gridpath, profilepaths[i])
    @assert get_logM(historypath) == get_logM(profilepath)
    logM = get_logM(historypath)
    if !isempty(only_use_these_logMs) if logM ∉ only_use_these_logMs; continue; end; end
    history_dual, profiles_dual = de.bookkeeping(historypath, profilepath,3)
    logM_dual = ForwardDiff.Dual{}(logM, 1.0,0.0,0.0)
    initial_params = [logM_dual, X_dual, Z_dual, Dfraction_dual, R_dual]
    model = de.Model_constructor(history_dual, profiles_dual, initial_params, inititial_params_names)
    track = nothing
    try 
        X_init = 0.4; X_end = 0.0000001
        X_init = 0.99*model.history_value.X_center[1]; X_end = 0.01
        track = de.Track(model,X_init, X_end, 1000)
    catch 
        println(" Track FAILED for logM = $logM")
        continue
    end
    println(" Track OK for logM = $logM")
    modeltracks[logM] = track; models[logM] = model
    push!(all_logMs, logM)
end
##
model = models[0.0]
Xzams = 0.995*model.history_value.X_center[1]
logL = log10(de.param1_to_param2(Xzams, model.history, "X_center","L_surf"))
de.find_index(Xzams, model.history, "X_center")
model.history.X_center[538]
log10(model.history.L_surf[538])
logT = log10(de.param1_to_param2(Xzams, model.history, "X_center","T_surf"))
de.find_index(Xzams, model.history, "X_center")
model.history.X_center[538]
log10(model.history.T_surf[538])
##
fig = Figure(size=(1000,500));
ax1 = Axis(fig[1,1],xreversed=true,xlabel = L"Central hydrogen fraction $X$",ylabel=L"\log L/L_\odot")
range = 534:540
scatter!(ax1, model.history_value.X_center[range], log10.(model.history_value.L_surf[range]),markersize=15,label="JEMS results",color=:lightgreen)
scatter!(ax1, model.history_value.X_center[538], log10(model.history_value.L_surf[538]),markersize=15,label="JEMS Model 538",color=:black, marker=:xcross)
vlines!(ax1,Xzams,color=:black,linestyle=:dash,label=L"X_{\text{ZAMS}}")
axislegend(ax1,position=:lt)
fig
##
N = length(models)
logMs = zeros(N); 
logL_val_zams = zeros(N); logL_partial_zams = zeros(N); logL_zams = zeros(typeof(Dual(1.0,1.0,1.0,1.0)),N)
logL_val_tams = zeros(N); logL_partial_tams = zeros(N); logL_tams = zeros(typeof(Dual(1.0,1.0,1.0,1.0)),N)
logL_val_pms =  zeros(N); logL_partial_pms =  zeros(N); logL_pms  = zeros(typeof(Dual(1.0,1.0,1.0,1.0)),N)
for ( (logM,model), i) in zip(models, 1:N)
    X_zams = 0.999*model.history_value.X_center[1]
    logL = log10(de.param1_to_param2(X_zams, model.history, "X_center","L_surf"))
    logMs[i] = logM; 
    logL_val_zams[i] = logL.value; 
    logL_partial_zams[i] = logL.partials[1]; 
    @show logL
    logL_zams[i] = logL

    X_tams = 0.0001
    logL = log10(de.param1_to_param2(X_tams, model.history, "X_center","L_surf"))
    logL_val_tams[i] = logL.value; logL_partial_tams[i] = logL.partials[1]; logL_tams[i] = logL

    T_pms = 5000
    logL = log10(de.param1_to_param2(T_pms, model.history, "T_surf","L_surf"))
    logL_val_pms[i] = logL.value; logL_partial_pms[i] = logL.partials[1]; logL_pms[i] = logL
end

## ###### FIGUUR MASS LUMINOSITY 
fig = Figure(size=(1300,1000));
ax1 = Axis(fig[1,1],xlabel=L"\log M/M_\odot",ylabel=L"\log L/L_\odot")
ax1.xlabelvisible=false; ax1.xticklabelsvisible=false
scatter!(ax1, logMs, logL_val_pms,markersize=15,label="PMS",color=:blue, marker=:star8)
scatter!(ax1, logMs, logL_val_zams,markersize=15,label="ZAMS",color=:lightgreen)
scatter!(ax1, logMs, logL_val_tams,markersize=15,label="TAMS",color=:black, marker=:xcross)
#scatter!(ax1, logMs, (d->d.partials[2]).(logLs))
ax2 = Axis(fig[2,1],ylabel=L"\frac{\partial \log L}{\partial \log M}")
ax2.xlabelvisible=false; ax2.xticklabelsvisible=false
scatter!(ax2, logMs, logL_partial_zams,markersize=25,label="ZAMS",color=:lightgreen)
scatter!(ax2, logMs, logL_partial_tams,markersize=15,label="TAMS",color=:black, marker=:xcross)
scatter!(ax2, logMs, logL_partial_pms,markersize=15,label="PMS",color=:blue, marker=:star8)
linkxaxes!(ax1, ax2)
axislegend(ax1, position=:lt)#; axislegend(ax2, position=:lt)
ax3 = Axis(fig[3,1],ylabel=L"\frac{\partial \log L}{\partial X_{\text{in}}}")
linkxaxes!(ax1, ax3); ax3.xlabelvisible=false; ax3.xticklabelsvisible=false
scatter!(ax3, logMs, (d->d.partials[2]).(logL_zams),markersize=15,label="ZAMS",color=:lightgreen)
scatter!(ax3, logMs, (d->d.partials[2]).(logL_tams),markersize=15,label="TAMS",color=:black, marker=:xcross)
scatter!(ax3, logMs, (d->d.partials[2]).(logL_pms),markersize=15,label="PMS",color=:blue, marker=:star8)
ax4 = Axis(fig[4,1],xlabel=L"\log M/M_\odot",ylabel=L"\frac{\partial \log L}{\partial Z_{\text{in}}}")
linkxaxes!(ax1, ax4)
scatter!(ax4, logMs, (d->d.partials[3]).(logL_zams),markersize=15,label="ZAMS",color=:lightgreen)
scatter!(ax4, logMs, (d->d.partials[3]).(logL_tams),markersize=15,label="TAMS",color=:black, marker=:xcross)
scatter!(ax4, logMs, (d->d.partials[3]).(logL_pms),markersize=15,label="PMS",color=:blue, marker=:star8)
fig
##save fig
savepath = "Figures/dual_mass_luminosity.png"
save(savepath, fig, px_per_unit=5); @show savepath
##

## ###### FIGUUR MASS LUMINOSITY  WITH 5 PARTIALS!
fig = Figure(size=(1200,1200));
ax1 = Axis(fig[1,1],xlabel=L"\log M/M_\odot",ylabel=L"\log L/L_\odot")
ax1.xlabelvisible=false; ax1.xticklabelsvisible=false
scatter!(ax1, logMs, logL_val_pms,markersize=15,label="PMS",color=:blue, marker=:star8)
scatter!(ax1, logMs, logL_val_zams,markersize=15,label="ZAMS",color=:lightgreen)
scatter!(ax1, logMs, logL_val_tams,markersize=15,label="TAMS",color=:black, marker=:xcross)
#scatter!(ax1, logMs, (d->d.partials[2]).(logLs))
ax2 = Axis(fig[2,1],ylabel=L"\partial \log L/\partial \log M")
ax2.xlabelvisible=false; ax2.xticklabelsvisible=false
scatter!(ax2, logMs, logL_partial_zams,markersize=25,label="ZAMS",color=:lightgreen)
scatter!(ax2, logMs, logL_partial_tams,markersize=15,label="TAMS",color=:black, marker=:xcross)
scatter!(ax2, logMs, logL_partial_pms,markersize=15,label="PMS",color=:blue, marker=:star8)
linkxaxes!(ax1, ax2)
axislegend(ax1, position=:lt)#; axislegend(ax2, position=:lt)
ax3 = Axis(fig[3,1],ylabel=L"\partial \log L / \partial X_{\text{in}}")
linkxaxes!(ax1, ax3); ax3.xlabelvisible=false; ax3.xticklabelsvisible=false
scatter!(ax3, logMs, (d->d.partials[2]).(logL_zams),markersize=15,label="ZAMS",color=:lightgreen)
scatter!(ax3, logMs, (d->d.partials[2]).(logL_tams),markersize=15,label="TAMS",color=:black, marker=:xcross)
scatter!(ax3, logMs, (d->d.partials[2]).(logL_pms),markersize=15,label="PMS",color=:blue, marker=:star8)
ax4 = Axis(fig[4,1],ylabel=L"\partial \log L / \partial Z_{\text{in}}")
linkxaxes!(ax1, ax4); ax4.xlabelvisible=false; ax4.xticklabelsvisible=false
scatter!(ax4, logMs, (d->d.partials[3]).(logL_zams),markersize=15,label="ZAMS",color=:lightgreen)
scatter!(ax4, logMs, (d->d.partials[3]).(logL_tams),markersize=15,label="TAMS",color=:black, marker=:xcross)
scatter!(ax4, logMs, (d->d.partials[3]).(logL_pms),markersize=15,label="PMS",color=:blue, marker=:star8)

ax4 = Axis(fig[5,1],xlabel=L"\log M/M_\odot",ylabel=L"\partial \log L / \partial D_{f,\text{in}}")
linkxaxes!(ax1, ax4); ax4.xlabelvisible=false; ax4.xticklabelsvisible=false
scatter!(ax4, logMs, (d->d.partials[4]).(logL_zams),markersize=15,label="ZAMS",color=:lightgreen)
scatter!(ax4, logMs, (d->d.partials[4]).(logL_tams),markersize=15,label="TAMS",color=:black, marker=:xcross)
scatter!(ax4, logMs, (d->d.partials[4]).(logL_pms),markersize=15,label="PMS",color=:blue, marker=:star8)

ax5 = Axis(fig[6,1],xlabel=L"\log M/M_\odot",ylabel=L"\partial \log L / \partial R_{\text{in}}")
linkxaxes!(ax1, ax5);
scatter!(ax5, logMs, (d->d.partials[5]).(logL_zams),markersize=15,label="ZAMS",color=:lightgreen)
scatter!(ax5, logMs, (d->d.partials[5]).(logL_tams),markersize=15,label="TAMS",color=:black, marker=:xcross)
scatter!(ax5, logMs, (d->d.partials[5]).(logL_pms),markersize=15,label="PMS",color=:blue, marker=:star8)

fig
##save fig
save("Figures/dual_mass_luminosity.png", fig, px_per_unit=5)
##


#########################################################################
## eerste look at JEMS dual results
historypath = "DualRuns/DualGrid/logM_0.0_X_0.7381_.history.hdf5";
profilepath = "DualRuns/DualGrid/logM_0.0_X_0.7381_.profiles.hdf5";

history, _ = de.bookkeeping(historypath, profilepath,5);
history_value = (d->d.value).(history)
log10(history.T_surf[500])
log10(history.L_surf[500])

de.find_index(0.99*history.X_center[1], history, "X_center")
log10(history.L_surf[540])
log10(de.param1_to_param2(0.99*history.X_center[1], history, "X_center","L_surf"))

de.find_index(0.99*history.X_center[1], history, "X_center")
history.X_center[540]
de.param1_to_param2(0.99*history_value.X_center[1], history, "X_center","X_center")
de.param1_to_param2(0.5, history, "X_center","X_center")


de.param1_to_param2(3000,history,"T_surf","T_surf")
de.param1_to_param2(20,history,"L_surf","L_surf")
de.param1_to_param2(0.5, history, "Y_center","Y_center")

de.find_index(0.999*history.X_center[1], history, "X_center")
log10(history.T_surf[533])
log10(de.param1_to_param2(0.999*history.X_center[1], history, "X_center","T_surf"))
###################################################################################################