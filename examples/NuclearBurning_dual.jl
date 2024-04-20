#=
# NuclearBurning_dual.jl

This example has the same structure as the NuclearBurning.jl example, 
but uses Dual numbers to compute partial derivatives with respect
ect to input parameters.
=#
using BenchmarkTools
using Jems.Chem
using Jems.Constants
using Jems.EOS
using Jems.Opacity
using Jems.NuclearNetworks
using Jems.Turbulence
using Jems.StellarModels
using Jems.Evolution
using Jems.ReactionRates
using Jems.DualSupport
using ForwardDiff


#=
Creation of external tag for the dual numbers

We make a tag for EXTERNAL use, i.e. to keep track of the derivatives with respect
to the input parameters which we will choose. The code itself will make an
internal tag for internal use.
On a computing level, the external tag is the 'inner tag' in the code, 
because these derivatives are taken along the entire ride through the code, 
while the dual numbers with internal tags are only used at specific points 
for local use and then dropped again.
=#
tag_external = ForwardDiff.Tag{:external, nothing}
ForwardDiff.tagcount(tag_external); #this function is necessary to order the tags
#create a dummy dual with the same dimensions as the input dual numbers
#e.g. 6 = 1 value + 5 partial derivatives


### Model creation, as usual

println("Create StellarModel ############################")
varnames = [:lnρ, :lnT, :lnr, :lum]
structure_equations = [Evolution.equationHSE, Evolution.equationT,
                       Evolution.equationContinuity, Evolution.equationLuminosity]
remesh_split_functions = [StellarModels.split_lnr_lnρ, StellarModels.split_lum,
                          StellarModels.split_lnT, StellarModels.split_xa]
net = NuclearNetwork([:H1, :He4, :C12, :N14, :O16], [(:kipp_rates, :kipp_pp), (:kipp_rates, :kipp_cno)])
nz = 1000
nextra = 100
eos = EOS.IdealEOS(true)
opacity = Opacity.SimpleElectronScatteringOpacity()
turbulence = Turbulence.BasicMLT(1.0)
dual_type = typeof(ForwardDiff.Dual{tag_external}(5.0,0.0,0.0,0.0,0.0,0.0))
number_of_partials = 5 #number of partials we want to keep track of
dual_type = ForwardDiff.Dual{tag_external,Float64,number_of_partials}
sm = StellarModel(varnames, structure_equations, nz, nextra, remesh_split_functions, net, eos, opacity, turbulence, number_type = dual_type);

##
#=
### Initialize StellarModel and evaluate equations and jacobian

We do not have a working initial condition yet. We require pressure, temperature profiles. One simple available initial
condition is that of an n=1 polytrope. This sets the pressure and density and computes the temperature from the EOS. The
luminosity is initialized by assuming pure radiative transport for the temperature gradient produced by the polytrope.

The normal evolution loop will store the information at the end of the step into an attribute of type `StellarStepInfo`,
stored at `sm.esi` (_end step info_). After initializing our polytrope we can mimic that behavior by calling 
`set_end_step_info!(sm)`. We then 'cycle' this info into the information of a hypothetical previous step with
`cycle_step_info`, so now `sm.psi` contains our initial condition. Finally we call `set_start_step_info` to use `sm.psi`
(_previous step info_) to populate the information needed before the Newton solver in `sm.ssi` (_start step info_).
At last we are in position to evaluate the kequations and compute the Jacobian.
=#
println("Initialize StellarModel ############################")
n = 3
#define dual input numbers, all partial derivatives are with respect to the mass
X_dual         = ForwardDiff.Dual{tag_external}(0.7154,  1.0,0.0,0.0,0.0,0.0) #(value, derivative to mass)
Z_dual         = ForwardDiff.Dual{tag_external}(0.0142,  0.0,1.0,0.0,0.0,0.0)
Dfraction_dual = ForwardDiff.Dual{tag_external}(0.0,     0.0,0.0,1.0,0.0,0.0)
mass_dual      = ForwardDiff.Dual{tag_external}(1.0*MSUN,0.0,0.0,0.0,1.0,0.0)
R_dual         = ForwardDiff.Dual{tag_external}(100*RSUN,0.0,0.0,0.0,0.0,1.0)
StellarModels.n_polytrope_initial_condition!(n, sm, nz, X_dual,Z_dual,Dfraction_dual,Chem.abundance_lists[:ASG_09],mass_dual, R_dual; initial_dt=10 * SECYEAR)
StellarModels.evaluate_stellar_model_properties!(sm, sm.props)
Evolution.cycle_props!(sm);
StellarModels.copy_scalar_properties!(sm.start_step_props, sm.prv_step_props)
StellarModels.copy_mesh_properties!(sm, sm.start_step_props, sm.prv_step_props)  # or do StellarModels.remesher!(sm);
StellarModels.evaluate_stellar_model_properties!(sm, sm.start_step_props)
StellarModels.copy_scalar_properties!(sm.props, sm.start_step_props)
StellarModels.copy_mesh_properties!(sm, sm.props, sm.start_step_props)

##

#=
### Evolving our model
=#
open("example_options.toml", "w") do file
    write(file,
          """
          [remesh]
          do_remesh = true

          [solver]
          newton_max_iter_first_step = 1000
          initial_model_scale_max_correction = 0.5
          newton_max_iter = 30
          scale_max_correction = 0.1
          report_solver_progress = false

          [timestep]
          dt_max_increase = 1.5
          delta_R_limit = 0.01
          delta_Tc_limit = 0.01
          delta_Xc_limit = 0.005

          [termination]
          max_model_number = 5
          max_center_T = 1e8

          [plotting]
          do_plotting = false
          wait_at_termination = false
          plotting_interval = 1

          window_specs = ["HR", "TRho", "profile", "history"]
          window_layouts = [[1, 1],  # arrangement of plots
                            [1, 2],
                            [2, 1],
                            [2, 2]
                            ]

          profile_xaxis = 'mass'
          profile_yaxes = ['log10_T']
          profile_alt_yaxes = ['X','Y']

          history_xaxis = 'star_age'
          history_yaxes = ['R_surf']
          history_alt_yaxes = ['T_center']

          [io]
          history_interval = 1
          profile_interval = 1
          terminal_header_interval = 2
          terminal_info_interval = 100

          """)
end
print("Initialize & Evolve StellarModel ############################################################")
StellarModels.set_options!(sm.opt, "./example_options.toml")
rm(sm.opt.io.hdf5_history_filename; force=true)
rm(sm.opt.io.hdf5_profile_filename; force=true)
StellarModels.n_polytrope_initial_condition!(n, sm, nz, X_dual,Z_dual,Dfraction_dual,Chem.abundance_lists[:ASG_09],
                                            mass_dual, R_dual; initial_dt=10 * SECYEAR)
@time Evolution.do_evolution_loop!(sm);

## 
using DataFrames
"""
get_dual_profile_dataframe_from_hdf5(hdf5_filename, value_name, partials_names)

Returns a DataFrame object built filled with Dual numbers.
"""
function get_dual_profile_dataframe_from_hdf5(hdf5_filename, value_name, partials_names)
    value_dataframe = StellarModels.get_profile_dataframe_from_hdf5(hdf5_filename, value_name)
    partial_dataframes = [StellarModels.get_profile_dataframe_from_hdf5(hdf5_filename, partial_name) for partial_name in partials_names]
    df_dual = ForwardDiff.Dual.(value_dataframe, partial_dataframes...)
    return df_dual
end
 
profile_names = StellarModels.get_profile_names_from_hdf5("profiles.hdf5")
value_names = [name for name in profile_names if !occursin("dual", name)]
dual_names_unpacked = [name for name in profile_names if occursin("dual", name)]
nbPartials = Int(length(dual_names_unpacked)/length(value_names))
dual_names = [[dual_name for dual_name in dual_names_unpacked[lo:lo+nbPartials-1] ] for lo in 1:nbPartials:(length(dual_names_unpacked))] 
dual_names

profile_number = 5
bla = get_dual_profile_dataframe_from_hdf5("profiles.hdf5", value_names[profile_number], dual_names[profile_number])

StellarModels.get_profile_dataframe_from_hdf5("profiles.hdf5", value_names[5])



















##
#=
### Plotting with Makie

Now that our simulation is complete we can analyze the results. We make use of the Makie package for this. I'm not a fan
of the Makie defaults, so I adjust them. I normally also adjust the fonts to be consistent with \LaTeX, but I avoid that
here so we don't need to distribute those fonts together with Jems.
=#
using CairoMakie, LaTeXStrings, MathTeXEngine
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
#=
### Compare against polytropes

Below we see how the profile of the star compares to different polytropes. We make use of the facility tools to obtain
DataFrame objects out of the hdf5 output. In particular, `get_profile_names_from_hdf5` will provide the names of all 
profiles contained within the hdf5 file, while `get_profile_dataframe_from_hdf5` is used to obtain one DataFrame
corresponding to one stellar profile. The animation is constructed using the `Observable` type that makie provides. Note
that the zero points of the polytropes are arbitrary.
=#
profile_names = StellarModels.get_profile_names_from_hdf5("profiles.hdf5")

f = Figure();
ax = Axis(f[1, 1]; xlabel=L"\log_{10}(\rho/\mathrm{[g\;cm^{-3}]})", ylabel=L"\log_{10}(P/\mathrm{[dyn]})")

pname = Observable(profile_names[1])

profile = @lift(StellarModels.get_profile_dataframe_from_hdf5("profiles.hdf5", $pname))
#To see why this is done this way, see https://docs.makie.org/stable/explanations/nodes/index.html#problems_with_synchronous_updates
#the main issue is that remeshing changes the size of the arrays
log10_ρ = @lift($profile[!, "log10_ρ"])
log10_P = Observable(rand(length(log10_ρ.val)))

profile_line = lines!(ax, log10_ρ, log10_P; label="real profile")
xvals = LinRange(-13, 4, 100)
lines!(ax, xvals, (1 + 1 / 1) .* xvals .+ 20; label="n=1")
lines!(ax, xvals, (1 + 1 / (1.5)) .* xvals .+ 15; label="n=1.5")
lines!(ax, xvals, (1 + 1 / 3) .* xvals .+ 15; label="n=3")
axislegend(ax; position=:rb)

model_number_str = @lift("model number=$(parse(Int,$pname))")
profile_text = text!(ax, -10, 20; text=model_number_str)

record(f, "rho_P_evolution.gif", profile_names[1:end]; framerate=2) do profile_name
    profile = StellarModels.get_profile_dataframe_from_hdf5("profiles.hdf5", profile_name)
    log10_P.val = profile[!, "log10_P"]
    pname[] = profile_name
end
##





##
# ![Movie polytrope](./rho_P_evolution.gif)

##
#=
### Check nuclear burning

We see that the structure evolves towards an n=3 polytrope. Deviations near the core are due to the non-homogeneous
composition as hydrogen is burnt. We can similarly visualize how the hydrogen mass fraction changes in the simulation.
In here, only one frame shows the hydrogen that was burnt. To better visualize that you can adjust `profile_interval` in
the [IO](Evolution.md##Io.jl) options (and probably adjust the framerate).
=#
profile_names = StellarModels.get_profile_names_from_hdf5("profiles.hdf5")

f = Figure();
ax = Axis(f[1, 1]; xlabel=L"\mathrm{Mass}\;[M_\odot]", ylabel=L"X")

pname = Observable(profile_names[1])

profile = @lift(StellarModels.get_profile_dataframe_from_hdf5("profiles.hdf5", $pname))
mass = @lift($profile[!, "mass"])
X = Observable(rand(length(mass.val)))
model_number_str = @lift("model number=$(parse(Int,$pname))")

profile_line = lines!(ax, mass, X; label="real profile")
profile_text = text!(ax, 0.7, 0.0; text=model_number_str)

record(f, "X_evolution.gif", profile_names[1:end]; framerate=2) do profile_name
    profile = StellarModels.get_profile_dataframe_from_hdf5("profiles.hdf5", profile_name)
    X.val = profile[!, "X"]
    pname[] = profile_name
end

# ![Movie polytrope](./X_evolution.gif)

##
#=
### Plot a funny HR diagram

Finally, we can also access the history data of the simulation. We use this to plot a simple HR diagram. As our
microphysics are very simplistic, and the initial condition is not very physical, this looks a bit funny!
=#
f = Figure();
ax = Axis(f[1, 1]; xlabel=L"\log_{10}(T_\mathrm{eff}/[K])", ylabel=L"\log_{10}(L/L_\odot)", xreversed=true)
history = StellarModels.get_history_dataframe_from_hdf5("history.hdf5")
lines!(ax, log10.(history[!, "T_surf"]), log10.(history[!, "L_surf"]))
f

##
#=
### Perform some cleanup

Internally we want to prevent storing any of the hdf5 files into our git repos, so I remove them. You can also take
advantage of `julia` as a scripting language to post-process your simulation output in a similar way.
=#
rm("history.hdf5")
rm("profiles.hdf5")
rm("example_options.toml")
