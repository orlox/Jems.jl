using HDF5
using DataFrames
using Printf

export add_history_option, add_profile_option,
    history_output_units, history_output_functions,
    profile_output_units, profile_output_functions

const width = 9
const decimals = 4
const floatstr = "%#$width.$decimals" * "g "
const intstr = "%$width" * "i "
const line1fmt = Printf.Format(intstr * floatstr^6 * intstr * "\n")
const line2fmt = Printf.Format(floatstr^7 * intstr * "\n")
const header = """
        model     logdt      logL   logTeff     logPs     logρs    H_cntr     iters
         mass       age      logR     logTc     logPc     logρc   He_cntr     zones
    -------------------------------------------------------------------------------
    """

"""
    history_get_ind_vars_edge_value(sm::StellarModel, var_symbol::Symbol, edge::Symbol)

Returns the value of the independent variable `var_symbol` at either the surface or the center of the StellarModel `sm`.
`edge` can be either `:surface` or `:center`.
"""
function history_get_ind_vars_edge_value(sm::StellarModel, var_symbol::Symbol, edge::Symbol)
    if var_symbol ∉ sm.var_names
        throw(ArgumentError(":$var_symbol is not a valid independent variable"))
    end
    if edge == :center
        return sm.ind_vars[sm.vari[var_symbol]]
    elseif edge == :surface
        return sm.ind_vars[(sm.nz - 1) * sm.nvars + sm.vari[var_symbol]]
    else
        throw(ArgumentError("'edge' must be either :surface or :center. Instead received 
                :$edge"))
    end
end

const history_output_units::Dict{String, String} = Dict()
const history_output_functions::Dict{String, Function} = Dict()
function add_history_option(name, unit, func)
    if haskey(history_output_units, name)
        throw(ArgumentError("Key $name is already part of the history output options"))
    end
    history_output_units[name] = unit
    history_output_functions[name] = func
end
add_history_option("star_age", "year", sm -> sm.esi.time / SECYEAR)
add_history_option("dt", "year", sm -> sm.esi.dt / SECYEAR)
add_history_option("model_number", "unitless", sm -> sm.esi.model_number)
add_history_option("star_mass", "Msun", sm -> sm.esi.mstar / MSUN)

#surface properties
add_history_option("R_surf", "Rsun", sm -> exp(sm.esi.lnr[sm.nz]-log(RSUN)))
add_history_option("L_surf", "Lsun", sm -> sm.esi.L[sm.nz])
add_history_option("T_surf", "K", sm -> exp(sm.esi.lnT[sm.nz]))
add_history_option("P_surf", "dyne", sm -> exp(sm.esi.lnP[sm.nz]))
add_history_option("ρ_surf", "g*cm^-3", sm -> exp(sm.esi.lnρ[sm.nz]))
add_history_option("X_surf", "unitless", sm -> history_get_ind_vars_edge_value(sm, :H1, :surface))
add_history_option("Y_surf", "unitless", sm -> history_get_ind_vars_edge_value(sm, :He4, :surface))

#central properties
add_history_option("T_center", "K", sm -> exp(sm.esi.lnT[1]))
add_history_option("P_center", "dyne", sm -> exp(sm.esi.lnP[1]))
add_history_option("ρ_center", "g*cm^-3", sm -> exp(sm.ssi.lnρ[1]))
add_history_option("X_center", "unitless", sm -> history_get_ind_vars_edge_value(sm, :H1, :surface))
add_history_option("Y_center", "unitless", sm -> history_get_ind_vars_edge_value(sm, :He4, :surface))

"""
    profile_get_ind_vars_value(sm::StellarModel, var_symbol::Symbol, k::Int)

Returns the value of the variable Symbol `var_symbol` at cell number `k` of the StellarModel `sm`.
"""
function profile_get_ind_vars_value(sm::StellarModel, var_symbol::Symbol, k::Int)
    if var_symbol ∉ sm.var_names
        throw(ArgumentError(":$var_symbol is not a valid independent variable"))
    end
    return sm.ind_vars[(k - 1) * sm.nvars + sm.vari[var_symbol]]
end

const profile_output_units::Dict{String, String} = Dict()
const profile_output_functions::Dict{String, Function} = Dict()
function add_profile_option(name, unit, func)
    if haskey(profile_output_units, name)
        throw(ArgumentError("Key $name is already part of the history output options"))
    end
    profile_output_units[name] = unit
    profile_output_functions[name] = func
end

#general properties
add_profile_option("zone", "unitless", (sm, k) -> k)
add_profile_option("mass", "Msun", (sm, k) -> sm.esi.m[k] / MSUN)
add_profile_option("dm", "Msun", (sm, k) -> sm.esi.dm[k] / MSUN)

#thermodynamic properties
add_profile_option("log10_r", "log10(Rsun)", (sm, k) -> sm.esi.lnr[k] / RSUN * log10_e)
add_profile_option("log10_P", "log10(dyne)", (sm, k) -> sm.esi.lnP[k] * log10_e)
add_profile_option("log10_T", "log10(K)", (sm, k) -> sm.esi.lnT[k] * log10_e)
add_profile_option("log10_ρ", "log10_(g*cm^-3)", (sm, k) -> sm.esi.lnρ[k] * log10_e)
add_profile_option("luminosity", "Lsun", (sm, k) -> sm.esi.L[k])

#abundance
add_profile_option("X", "unitless", (sm, k) -> profile_get_ind_vars_value(sm, :H1, k))
add_profile_option("Y", "unitless", (sm, k) -> profile_get_ind_vars_value(sm, :He4, k))

"""
    create_output_files(sm::StellarModel)

Creates output files for history and profile data
"""
function create_output_files!(sm::StellarModel)
    # Create history file
    sm.history_file = h5open(sm.opt.io.hdf5_history_filename, "w")
    data_cols = sm.opt.io.history_values
    ncols = length(data_cols)

    # verify validity of column names
    for i in eachindex(data_cols)
        if data_cols[i] ∉ keys(history_output_functions)
            throw(ArgumentError("Invalid name for history data column, 
                :$(data_cols[i])"))
        end
    end

    # Create history dataset in HDF5 file
    # Dataset is created with size (0, ncols), we will add rows by using the HDF5.set_extent_dims function
    # the (-1, ncols) is used to define the maximum extent of the dataset, -1 indicates that it is unbound
    # in number of rows. The chunk size is used for compression. Smaller chunk sizes will result in worse
    # compression but faster writes.
    # The compression level can be anywhere between 0 and 9, 0 being no compression 9 being the highest.
    # Compression is lossless.
    history = create_dataset(sm.history_file, "history", Float64, ((0, ncols), (-1, ncols)),
                                chunk=(sm.opt.io.hdf5_history_chunk_size, ncols),
                                compress=sm.opt.io.hdf5_history_compression_level)

    # next up, include the units for all quantities. No need to recheck columns.
    attrs(history)["column_units"] = [history_output_units[data_cols[i]] for i in eachindex(data_cols)]
    # Finally, place column names
    attrs(history)["column_names"] = [data_cols[i] for i in eachindex(data_cols)]
    if (!sm.opt.io.hdf5_history_keep_open)
        close(sm.history_file)
    end

    # Create profile file
    sm.profiles_file = h5open(sm.opt.io.hdf5_profile_filename, "w")
    data_cols = sm.opt.io.profile_values
    # verify validity of column names
    for i in eachindex(data_cols)
        if data_cols[i] ∉ keys(profile_output_functions)
            throw(ArgumentError("Invalid name for history data column,
                :$(data_cols[i])"))
        end
    end
    if (!sm.opt.io.hdf5_profile_keep_open)
        close(sm.profiles_file)
    end
end

function close_output_files!(sm)
    if(sm.opt.io.hdf5_history_keep_open)
        close(sm.history_file)
    end
    if(sm.opt.io.hdf5_profile_keep_open)
        close(sm.profiles_file)
    end
end

"""
    write_data(sm::StellarModel)

Saves data (history/profile) for the current model, as required by the settings in `sm.opt.io`.
"""
function write_data(sm::StellarModel)
    # do history
    if (sm.opt.io.history_interval > 0)
        file_exists = isfile(sm.opt.io.hdf5_history_filename)
        if !file_exists  # create file if it doesn't exist yet
            throw(ErrorException("History file does not exist at $(sm.opt.io.hdf5_history_filename)"))
        end
        if (sm.model_number % sm.opt.io.history_interval == 0)
            if (!sm.opt.io.hdf5_history_keep_open)
                sm.history_file = h5open(sm.opt.io.hdf5_history_filename, "r+")
            end
            data_cols = sm.opt.io.history_values
            ncols = length(data_cols)

            # after being sure the header is there, print the data
            history = sm.history_file["history"]
            HDF5.set_extent_dims(history, (size(history)[1] + 1, ncols))
            for i in eachindex(data_cols)
                history[end, i] = history_output_functions[data_cols[i]](sm)
            end
            if (!sm.opt.io.hdf5_history_keep_open)
                close(sm.history_file)
            end
        end
    end
    # do profile
    if (sm.opt.io.profile_interval > 0)
        file_exists = isfile(sm.opt.io.hdf5_profile_filename)
        if !file_exists  # create file if it doesn't exist yet
            throw(ErrorException("Profile file does not exist at $(sm.opt.io.hdf5_profile_filename)"))
        end
        if (sm.model_number % sm.opt.io.profile_interval == 0)
            if (!sm.opt.io.hdf5_profile_keep_open)
                sm.profiles_file = h5open(sm.opt.io.hdf5_profile_filename, "r+")
            end
            data_cols = sm.opt.io.profile_values
            ncols = length(data_cols)
            # Save current profile
            profile = create_dataset(sm.profiles_file,
                                        "$(lpad(sm.model_number,sm.opt.io.hdf5_profile_dataset_name_zero_padding,"0"))",
                                        Float64, ((sm.nz, ncols), (sm.nz, ncols));
                                        chunk=(sm.opt.io.hdf5_profile_chunk_size, ncols),
                                        compress=sm.opt.io.hdf5_profile_compression_level)

            # next up, include the units for all quantities. No need to recheck columns.
            attrs(profile)["column_units"] = [profile_output_units[data_cols[i]] for i in eachindex(data_cols)]
            # Place column names
            attrs(profile)["column_names"] = [data_cols[i] for i in eachindex(data_cols)]

            # store data
            for i in eachindex(data_cols), k = 1:(sm.nz)
                profile[k, i] = profile_output_functions[data_cols[i]](sm, k)
            end
            if (!sm.opt.io.hdf5_profile_keep_open)
                close(sm.profiles_file)
            end
        end
    end
end

function write_terminal_info(sm::StellarModel)
    if sm.model_number == 1 || 
            sm.model_number % sm.opt.io.terminal_header_interval == 0
        print(header)
    end
    if sm.model_number == 1 ||
            sm.model_number % sm.opt.io.terminal_info_interval == 0
        Printf.format(stdout, line1fmt,
                      sm.model_number,
                      log10(sm.dt/SECYEAR),
                      log10(get_cell_value(sm.props.L[sm.props.nz])),
                      log10_e * get_cell_value(sm.props.lnT[sm.props.nz]),
                      log10(get_cell_value(sm.props.eos_res[sm.props.nz].P)),
                      log10_e * get_cell_value(sm.props.lnρ[sm.props.nz]),
                      get_cell_value(sm.props.xa[1, sm.network.xa_index[:H1]]),
                      sm.solver_data.newton_iters)
        Printf.format(stdout, line2fmt,
                      sm.mstar / MSUN,
                      sm.time / SECYEAR,
                      log10_e * get_cell_value(sm.props.lnr[sm.props.nz]) - log10(RSUN),
                      log10_e * get_cell_value(sm.props.lnT[1]),
                      log10(get_cell_value(sm.props.eos_res[1].P)), 
                      log10_e * get_cell_value(sm.props.lnρ[1]),
                      get_cell_value(sm.props.xa[1, sm.network.xa_index[:He4]]),
                      sm.props.nz)
        println()
    end
end


"""
    get_history_dataframe_from_hdf5(hdf5_filename)

Returns a DataFrame object built from an hdf5 file, named `hdf5_filename`.
"""
function get_history_dataframe_from_hdf5(hdf5_filename)
    h5open(hdf5_filename) do history_file
        return DataFrame(history_file["history"][:, :], attrs(history_file["history"])["column_names"])
    end
end

"""
    get_profile_names_from_hdf5(hdf5_filename)

Retruns the column names of the profile data contained in the hdf5 file `hdf5_filename`.
"""
function get_profile_names_from_hdf5(hdf5_filename)
    h5open(hdf5_filename) do profiles_file
        return keys(profiles_file)
    end
end

"""
    get_profile_dataframe_from_hdf5(hdf5_filename, profile_name)

Returns a DataFrame object built from an hdf5 file, named `hdf5_filename`, considering the column named `profile_name`
"""
function get_profile_dataframe_from_hdf5(hdf5_filename, profile_name)
    h5open(hdf5_filename) do profiles_file
        return DataFrame(profiles_file[profile_name][:, :], attrs(profiles_file[profile_name])["column_names"])
    end
end
