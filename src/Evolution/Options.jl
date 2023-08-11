using TOML

@kwdef mutable struct SolverOptions
    newton_max_iter::Int = 100
    newton_max_iter_first_step::Int = 5000
    initial_model_scale_max_correction::Float64 = 3.0
    scale_max_correction::Float64 = 0.5
    scale_correction_negative_Lsurf::Float64 = 0.1
end

@kwdef mutable struct TimestepOptions
    initial_dt::Int = 1 # in years

    delta_R_limit::Float64 = 0.005
    delta_Tc_limit::Float64 = 0.005
    delta_Xc_limit::Float64 = 0.005

    dt_max_increase::Float64 = 2
    dt_retry_decrease::Float64 = 2
end

@kwdef mutable struct TerminationOptions
    max_model_number::Int = 1
    max_center_T::Float64 = 1e99
end

@kwdef mutable struct IOOptions
    hdf5_history_filename::String = "history.hdf5"
    hdf5_history_chunk_size::Int = 50
    hdf5_history_compression_level::Int = 9

    hdf5_profile_filename::String = "profiles.hdf5"
    hdf5_profile_chunk_size::Int = 50
    hdf5_profile_compression_level::Int = 9
    hdf5_profile_dataset_name_zero_padding::Int = 10

    history_interval::Int = 1
    profile_interval::Int = 10

    history_values::Vector{String} = ["star_age", "dt", "model_number", "star_mass", "R_surf", "L_surf", "T_surf",
                                      "P_surf", "ρ_surf", "X_surf", "Y_surf", "T_center", "P_center", "ρ_center",
                                      "X_center", "Y_center"]

    profile_values::Vector{String} = ["zone", "mass", "dm", "log10_ρ", "log10_P", "log10_T", "luminosity", "X", "Y"]
end

mutable struct Options
    solver::SolverOptions
    timestep::TimestepOptions
    termination::TerminationOptions
    io::IOOptions

    function Options()
        new(SolverOptions(), TimestepOptions(), TerminationOptions(), IOOptions())
    end
end

function set_options!(opt::Options, toml_path::String)
    options_file = TOML.parse(open(toml_path))

    # Verify all keys in the option file are valid
    # Do this before anything is changed, in that way if the load will fail the
    # input is unmodified
    for key in keys(options_file)
        if !(key in ["solver", "timestep", "termination", "io"])
            throw(ArgumentError("Error while reading $toml_path. 
                    One of the sections on the TOML file provided ([$key]) is not valid."))
        end
    end
    for key in keys(options_file)
        section = getfield(opt, Symbol(key))
        for subkey in keys(options_file[key])
            # verify this is a valid option for this section
            if (!(Symbol(subkey) in fieldnames(typeof(section))))
                throw(ArgumentError("Error while reading $toml_path. Option $subkey is not valid for section [$key]."))
            end
            try
                setfield!(section, Symbol(subkey), options_file[key][subkey])
            catch e
                io = IOBuffer()
                showerror(io, e)
                error_message = String(take!(io))
                if isa(e, TypeError)
                    throw(ArgumentError("""
                            While reading $toml_path a type error was produced when setting $subkey in [$key].
                            Verify the type is consistent with $(typeof(getfield(section, Symbol(subkey)))).
                            Full error message is displayed below:\n\n""" * error_message))
                end
                # Exception is not clear, create an IO buffer to load the exception message
                throw(ArgumentError("""
                        Unknown error while reading $toml_path.
                        An exception of type $(typeof(e)) was produced while setting $subkey in [$key].
                        The original error from the exception is shown below:\n\n""" * error_message))
            end
        end
    end
end

opt = Options()
