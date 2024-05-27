#=
# OneZoneBurn.jl

This notebook provides a simple example of a single zone undergoing nuclear burning.
Just as in NuclearBurning.jl, we start by importing all necessary Jems modules.
=#
using Jems.NuclearNetworks
using Jems.StellarModels
using Jems.Evolution
using Jems.Constants
using Jems.DualSupport
using Jems.Chem
using BenchmarkTools


#=
### Model creation

We start by creating our OneZone model. Contrary to a fully-fledged StellarModel, we need only to specify the nuclear
net used, along with its reactions. We also provide a custom equation for the composition, that does not include mixing
(we don't need it, of course, there is only one zone in this model).
=#
net = NuclearNetwork([:H1, :D2, :He3, :He4, :T3,
:Be7, :Li7, :B8,
 :C12,   :C13,   
 :n1,
],
[
    (:jina_rates, :n1_to_H1_wc12_w_x_0),
    (:jina_rates, :n1_H1_to_D2_an06_n_x_0),
    (:jina_rates, :n1_H1_to_D2_an06_n_x_1),
    (:jina_rates, :n1_H1_to_D2_an06_n_x_2),

    (:jina_rates, :H1_D2_to_He3_de04_n_x_0),
    (:jina_rates, :H1_D2_to_He3_de04_x_x_0),

    (:jina_rates, :D2_D2_to_H1_T3_go17_n_x_0),
    (:jina_rates, :D2_D2_to_n1_He3_gi17_n_x_0),
    (:jina_rates, :D2_T3_to_n1_He4_de04_x_x_0),
    (:jina_rates, :D2_T3_to_n1_He4_de04_x_x_1),

    (:jina_rates, :D2_He3_to_H1_He4_de04_x_x_0),
    (:jina_rates, :D2_He3_to_H1_He4_de04_x_x_1),

    (:jina_rates, :He4_T3_to_Li7_de04_x_x_0),
    (:jina_rates, :He4_He3_to_Be7_cd08_n_x_0),
    (:jina_rates, :He4_He3_to_Be7_cd08_n_x_1),

    (:jina_rates, :n1_He3_to_H1_T3_de04_x_x_0),
    (:jina_rates, :n1_He3_to_H1_T3_de04_x_x_1),

    (:jina_rates, :n1_Be7_to_H1_Li7_db18_x_x_0),
    (:jina_rates, :H1_Li7_to_He4_He4_de04_x_x_0),
    (:jina_rates, :H1_Li7_to_He4_He4_de04_r_x_0),
    (:jina_rates, :H1_Li7_to_He4_He4_de04_x_x_1),
    (:jina_rates, :H1_Li7_to_He4_He4_de04_r_x_1),
    ]


)

function equation_composition(oz::OneZone, k::Int, iso_name::Symbol)  # needs this signature for TypeStableEquations
    # Get mass fraction for this iso
    X00 = get_00_dual(oz.props.xa[oz.network.xa_index[iso_name]])
    dXdt_nuc::typeof(X00) = 0
    reactions_in = oz.network.species_reactions_in[oz.network.xa_index[iso_name]]
    for reaction_in in reactions_in
        rate = get_00_dual(oz.props.rates[reaction_in[1]])
        dXdt_nuc -= rate * reaction_in[2] * Chem.isotope_list[iso_name].A * AMU
    end
    reactions_out = oz.network.species_reactions_out[oz.network.xa_index[iso_name]]
    for reaction_out in reactions_out
        rate = get_00_dual(oz.props.rates[reaction_out[1]])
        dXdt_nuc += rate * reaction_out[2] * Chem.isotope_list[iso_name].A * AMU
    end
    Xi = get_value(oz.prv_step_props.xa[oz.network.xa_index[iso_name]])  # is never a dual!!
    return ((X00 - Xi) / oz.props.dt - dXdt_nuc)
end
oz = OneZone(equation_composition, net);

##
#=
### Set initial conditions
We now provide the initial conditions of our One Zone model, setting the temperature, density, starting dt and age, and
the initial abundances for all isotopes defined in our network above.
=#
oz.props.T = 1e7  # K
oz.props.ρ = 1  # g cm^{-3}
oz.props.ind_vars = zeros(oz.network.nspecies)
# oz.props.ind_vars[oz.network.xa_index[:H1]] = 1.0 # for full hydrogen mixture
# mass_fractions = get_mass_fractions(Chem.abundance_lists[:ASG_09],
#                                         oz.network.species_names, 0.7, 0.02, 0.0) 
for name in oz.network.species_names
    oz.props.ind_vars[oz.network.xa_index[name]] = 0 # mass_fractions[name]
end
oz.props.ind_vars[oz.network.xa_index[:n1]] = 0.15 # for full hydrogen mixture
oz.props.ind_vars[oz.network.xa_index[:H1]] = 0.85 # for full hydrogen mixture
# oz.props.ind_vars[oz.network.xa_index[:D2]] = 1e-10 # for full hydrogen mixture


oz.props.dt_next = 1 * SECYEAR
oz.props.time = 0.0
oz.props.model_number = 0

open("example_options.toml", "w") do file
    write(file,
          """

          [solver]
          newton_max_iter_first_step = 1000
          initial_model_scale_max_correction = 0.2
          newton_max_iter = 50
          scale_max_correction = 0.2
          report_solver_progress = false
          solver_progress_iter = 50

          [timestep]
          dt_max_increase = 1.5
          delta_Xc_limit = 0.005

          [termination]
          max_model_number = 200

          [plotting]
          do_plotting = false
          wait_at_termination = false

          plotting_interval = 1

          window_specs = ["history"]
          window_layout = [[1, 1]]
          yaxes_log = [true]
          
          history_xaxis = "age"
          history_yaxes = ["H1", "D2", "He3", "He4"]

          [io]
          profile_interval = 50
          terminal_header_interval = 100
          terminal_info_interval = 100
          history_values = ["age", "dt", "model_number", "T", "ρ", "n1",
                            "H1", "D2", "He3", "He4", "Li7", "Be7"]

          """)
end
StellarModels.set_options!(oz.opt, "./example_options.toml")
Evolution.do_one_zone_burn!(oz)

##

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
# GLMakie.activate!()

### Plot the history

history = StellarModels.get_history_dataframe_from_hdf5("history.hdf5")
# H1_pos = filter(x -> x > 0, history[!, "H1"])
# H1_x = history[!, "age"][1:length(H1_pos)]

function take_pos(x_list, y_list)
    y_new = filter(x -> x > 0, y_list)
    x_new = x_list[1:length(y_new)]
    return x_new, y_new
end

f = Figure();
ax = Axis(f[1, 1]; xlabel="age [yr]", ylabel=L"\log_{10}\mathrm{X}", xscale=log10, xminorticks=IntervalsBetween(10, false))
# history = StellarModels.get_history_dataframe_from_hdf5("history.hdf5")

lines!(ax, take_pos(history[!, "age"], history[!, "H1"])[1], log10.(take_pos(history[!, "age"], history[!, "H1"])[2]), label=L"^1\mathrm{H}")
lines!(ax, take_pos(history[!, "age"], history[!, "n1"])[1], log10.(take_pos(history[!, "age"], history[!, "n1"])[2]), label=L"^1\mathrm{n}")
lines!(ax, take_pos(history[!, "age"], history[!, "D2"])[1], log10.(take_pos(history[!, "age"], history[!, "D2"])[2]), label=L"^2\mathrm{D}")
lines!(ax, take_pos(history[!, "age"], history[!, "He3"])[1], log10.(take_pos(history[!, "age"], history[!, "He3"])[2]), label=L"^3\mathrm{He}")

# lines!(ax, history[!, "age"], log10.(history[!, "n1"]), label=L"^1\mathrm{n}")
# lines!(ax, history[!, "age"], log10.(history[!, "D2"]), label=L"^2\mathrm{H}")
# lines!(ax, history[!, "age"], log10.(history[!, "He3"]), label=L"^3\mathrm{He}")
# lines!(ax, history[!, "age"], log10.(history[!, "He4"]), label=L"^4\mathrm{He}")
# lines!(ax, history[!, "age"], log10.(history[!, "Li7"]), label=L"^7\mathrm{Li}")
# lines!(ax, history[!, "age"], log10.(history[!, "Be7"]), label=L"^7\mathrm{Be}")
# lines!(ax, history[!, "age"], log10.(history[!, "Be9"]), label=L"^9\mathrm{Be}")
# lines!(ax, history[!, "age"], log10.(history[!, "C12"]), label=L"^{12}\mathrm{C}")
# lines!(ax, history[!, "age"], log10.(history[!, "N14"]), label=L"^{14}\mathrm{N}")
# lines!(ax, history[!, "age"], log10.(history[!, "O16"]), label=L"^{16}\mathrm{O}")
# lines!(ax, history[!, "age"], log10.(history[!, "Ne20"]), label=L"^{20}\mathrm{Ne}")
axislegend(position=:lb)
ylims!(ax, -15,0.1)
xlims!(ax, 1, 1e20)
f

# save("OZ_PP_CNO_HB_Full.png", f)

##
#=
### Perform some cleanup

Internally we want to prevent storing any of the hdf5 files into our git repos, so I remove them. You can also take
advantage of `julia` as a scripting language to post-process your simulation output in a similar way.
=#
rm("history.hdf5")
rm("profiles.hdf5")
rm("example_options.toml")