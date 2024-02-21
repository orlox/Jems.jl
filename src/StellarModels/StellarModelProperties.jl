using ForwardDiff

abstract type AbstractStellarModelProperties end

@kwdef mutable struct StellarModelProperties{TN, TDual, TCellDualData, TFaceDualData} <: AbstractStellarModelProperties
    # scalar quantities
    dt::TN  # Timestep of the current evolutionary step (s)
    time::TN  # Age of the model (s)
    model_number::Int
    mstar::TN  # Total model mass (g)

    # array of the values of the independent variables, everything should be reconstructable from this (mesh dependent)
    ind_vars::Vector{TN}
    nz::Int
    m::Vector{TN}
    dm::Vector{TN}

    eos_res_dual::Vector{EOSResults{TDual}}
    eos_res::Vector{EOSResults{TCellDualData}}

    # independent variables (duals constructed from the ind_vars array)
    # represents a staggered mesh: T, ρ and abundances are defined in the center of each cell, L and r on the outer face
    lnT::Vector{TCellDualData}  # [K]
    lnρ::Vector{TCellDualData}  # [g cm^-3]
    lnr::Vector{TCellDualData}  # [cm]
    L::Vector{TCellDualData}    # erg g^-1
    xa::Matrix{TCellDualData}   # dim-less
    xa_dual::Matrix{TDual}      # only the cell duals wrt to itself

    # opacity (cell centered)
    κ::Vector{TCellDualData}  # cm^2 g^-1

    # rates (cell centered)
    rates::Matrix{TCellDualData}  # 
    rates_dual::Matrix{TDual}

    # face values
    lnP_face::Vector{TFaceDualData}  # [dyne]
    lnT_face::Vector{TFaceDualData}  # [K]
    κ_face::Vector{TFaceDualData}    # cm^2 g^-1
    ∇ₐ_face::Vector{TFaceDualData}   # dim-less
    ∇ᵣ_face::Vector{TFaceDualData}   # dim-less
end

function StellarModelProperties(nvars::Int, nz::Int, nextra::Int,
                                nrates::Int, nspecies::Int, vari::Dict{Symbol, Int},
                                ::Type{TN}) where {TN<:Real}

    # define the types
    CDDTYPE = CellDualData{nvars+1,3*nvars+1,TN}  # full dual arrays
    FDDTYPE = FaceDualData{2*nvars+1,3*nvars+1,TN}
    TD = typeof(ForwardDiff.Dual(zero(TN), (zeros(TN, nvars))...))  # only the cell duals

    # create the vector containing the independent variables
    ind_vars = zeros(TN, nvars * (nz + nextra))

    eos_res_dual = [EOSResults{TD}() for i in 1:(nz+nextra)]
    eos_res = [EOSResults{CDDTYPE}() for i in 1:(nz+nextra)]

    lnT = [CellDualData(nvars, TN; is_ind_var=true, ind_var_i=vari[:lnT]) for i in 1:(nz+nextra)]
    lnρ = [CellDualData(nvars, TN; is_ind_var=true, ind_var_i=vari[:lnρ]) for i in 1:(nz+nextra)]
    lnr = [CellDualData(nvars, TN; is_ind_var=true, ind_var_i=vari[:lnr]) for i in 1:(nz+nextra)]
    L = [CellDualData(nvars, TN; is_ind_var=true, ind_var_i=vari[:lum]) for i in 1:(nz+nextra)]
    xa = Matrix{CDDTYPE}(undef,nz+nextra, nspecies)
    for k in 1:(nz+nextra)
        for i in 1:nspecies
            xa[k,i] = CellDualData(nvars, TN;
                        is_ind_var=true, ind_var_i=4+i)
        end
    end
    xa_dual = zeros(TD, nz+nextra, nspecies)
    rates_dual = zeros(TD, nz+nextra, nrates)
    m = zeros(TN, nz+nextra)
    dm = zeros(TN, nz+nextra)

    # for some reason using zeros just creates a bunch of instances of the same object
    # so we just initialize a vector of undef
    lnP_face = Vector{FDDTYPE}(undef, nz+nextra)#zeros(FDDTYPE, nz+nextra)
    lnT_face = Vector{FDDTYPE}(undef, nz+nextra)#zeros(FDDTYPE, nz+nextra)
    κ_face = Vector{FDDTYPE}(undef, nz+nextra)#zeros(FDDTYPE, nz+nextra)
    ∇ₐ_face = Vector{FDDTYPE}(undef, nz+nextra)#zeros(FDDTYPE, nz+nextra)
    ∇ᵣ_face = Vector{FDDTYPE}(undef, nz+nextra)#zeros(FDDTYPE, nz+nextra)
    κ = Vector{CDDTYPE}(undef, nz+nextra)  # zeros(CDDTYPE, nz+nextra)
    for k in 1:(nz+nextra)
        lnP_face[k] = FaceDualData(nvars, TN)
        lnT_face[k] = FaceDualData(nvars, TN)
        κ_face[k] = FaceDualData(nvars, TN)
        ∇ₐ_face[k] = FaceDualData(nvars, TN)
        ∇ᵣ_face[k] = FaceDualData(nvars, TN)
        κ[k] = CellDualData(nvars, TN)
    end

    rates = Matrix{CDDTYPE}(undef, nz+nextra, nrates)
    for k in 1:(nz+nextra)
        for i in 1:nrates
            rates[k,i] = CellDualData(nvars, TN)
        end
    end

    return StellarModelProperties(;ind_vars=ind_vars, model_number=zero(Int),
                                  nz=nz, m=m, dm=dm, mstar=zero(TN),
                                  dt=zero(TN), time=zero(TN),
                                  eos_res_dual=eos_res_dual,
                                  eos_res=eos_res,
                                  lnT=lnT,
                                  lnρ=lnρ,
                                  lnr=lnr,
                                  L=L,
                                  xa=xa,
                                  xa_dual=xa_dual,
                                  lnP_face=lnP_face,
                                  lnT_face=lnT_face,
                                  κ_face=κ_face,
                                  ∇ₐ_face=∇ₐ_face,
                                  ∇ᵣ_face=∇ᵣ_face,
                                  κ=κ,
                                  rates=rates,
                                  rates_dual=rates_dual)
end

"""
    function evaluate_stellar_model_properties!(sm, props::StellarModelProperties{TDual, TCellDualData}) where
        {TDual <: ForwardDiff.Dual, TCellDualData}

Evaluates the stellar model properties `props` from the `ind_vars` array. The goal is to save the 'state' of the
StellarModel so we can easily get properties like rates, eos, opacity values, and retrace if a retry is called.
This does _not_ update the mesh/ind_vars arrays.
"""
function evaluate_stellar_model_properties!(sm, props::StellarModelProperties{TN, TDual, TCellDualData}) where
                                                {TN<:Real, TDual<:ForwardDiff.Dual, TCellDualData}
    lnT_i = sm.vari[:lnT]
    lnρ_i = sm.vari[:lnρ]
    lnr_i = sm.vari[:lnr]
    L_i = sm.vari[:lum]

    Threads.@threads for i in 1:props.nz
        # update independent variables
        update_cell_dual_data_value!(props.lnT[i], props.ind_vars[(i-1)*(sm.nvars)+lnT_i])
        update_cell_dual_data_value!(props.lnρ[i], props.ind_vars[(i-1)*(sm.nvars)+lnρ_i])
        update_cell_dual_data_value!(props.lnr[i], props.ind_vars[(i-1)*(sm.nvars)+lnr_i])
        update_cell_dual_data_value!(props.L[i], props.ind_vars[(i-1)*(sm.nvars)+L_i])
        for j in 1:sm.network.nspecies
            update_cell_dual_data_value!(props.xa[i,j],
                            props.ind_vars[(i-1)*(sm.nvars)+(sm.nvars - sm.network.nspecies + j)])
            props.xa_dual[i,j] = get_cell_dual(props.xa[i,j])
        end

        lnT = get_cell_dual(props.lnT[i])
        lnρ = get_cell_dual(props.lnρ[i])
        xa = @view props.xa_dual[i,:]

        # evaluate EOS
        set_EOS_resultsTρ!(sm.eos, props.eos_res_dual[i], lnT, lnρ,
                            xa, sm.network.species_names)
        #names = fieldnames(EOSResults)
        #for name in names
        #    dual = getfield(props.eos_res_dual[i], name)
        #    dual_cell_data = getfield(props.eos_res[i], name)
        #    update_cell_dual_data!(dual_cell_data, dual)
        #end
        update_cell_dual_data!(props.eos_res[i].T, props.eos_res_dual[i].T)
        update_cell_dual_data!(props.eos_res[i].P, props.eos_res_dual[i].P)
        update_cell_dual_data!(props.eos_res[i].ρ, props.eos_res_dual[i].ρ)
        update_cell_dual_data!(props.eos_res[i].lnT, props.eos_res_dual[i].lnT)
        update_cell_dual_data!(props.eos_res[i].lnP, props.eos_res_dual[i].lnP)
        update_cell_dual_data!(props.eos_res[i].lnρ, props.eos_res_dual[i].lnρ)
        update_cell_dual_data!(props.eos_res[i].Prad, props.eos_res_dual[i].Prad)
        update_cell_dual_data!(props.eos_res[i].μ, props.eos_res_dual[i].μ)
        update_cell_dual_data!(props.eos_res[i].α, props.eos_res_dual[i].α)
        update_cell_dual_data!(props.eos_res[i].β, props.eos_res_dual[i].β)
        update_cell_dual_data!(props.eos_res[i].δ, props.eos_res_dual[i].δ)
        update_cell_dual_data!(props.eos_res[i].χ_ρ, props.eos_res_dual[i].χ_ρ)
        update_cell_dual_data!(props.eos_res[i].χ_T, props.eos_res_dual[i].χ_T)
        update_cell_dual_data!(props.eos_res[i].u, props.eos_res_dual[i].u)
        update_cell_dual_data!(props.eos_res[i].cₚ, props.eos_res_dual[i].cₚ)
        update_cell_dual_data!(props.eos_res[i].∇ₐ, props.eos_res_dual[i].∇ₐ)
        update_cell_dual_data!(props.eos_res[i].Γ₁, props.eos_res_dual[i].Γ₁)

        # evaluate opacity
        κ_dual = get_opacity_resultsTρ(sm.opacity, lnT, lnρ,
                    xa, sm.network.species_names)
        update_cell_dual_data!(props.κ[i], κ_dual)

        # evaluate rates
        rates = @view props.rates_dual[i,:]
        set_rates_for_network!(rates, sm.network, props.eos_res_dual[i], xa)
        for j in eachindex(rates)
            update_cell_dual_data!(props.rates[i,j], rates[j])
        end
    end

    # do face values next
    Threads.@threads for i in 1:props.nz - 1
        κ00 = get_face_00_dual(props.κ[i])
        κp1 = get_face_p1_dual(props.κ[i+1])
        κface_dual = exp((props.dm[i] * log(κ00) + props.dm[i + 1] * log(κp1)) / (props.dm[i] + props.dm[i + 1]))
        update_face_dual_data!(props.κ_face[i], κface_dual)

        lnP₀ = get_face_00_dual(props.eos_res[i].lnP)
        lnP₊ = get_face_p1_dual(props.eos_res[i+1].lnP)
        lnP_face_dual = (props.dm[i] * lnP₀ + props.dm[i + 1] * lnP₊)/(props.dm[i] + props.dm[i + 1])
        update_face_dual_data!(props.lnP_face[i], lnP_face_dual)

        lnT₀ = get_face_00_dual(props.eos_res[i].lnT)
        lnT₊ = get_face_p1_dual(props.eos_res[i+1].lnT)
        lnT_face_dual = (props.dm[i] * lnT₀ + props.dm[i + 1] * lnT₊)/(props.dm[i] + props.dm[i + 1])
        update_face_dual_data!(props.lnT_face[i], lnT_face_dual)

        ∇ₐ_00 = get_face_00_dual(props.eos_res[i].∇ₐ)
        ∇ₐ_p1 = get_face_p1_dual(props.eos_res[i+1].∇ₐ)
        ∇ₐ_face_dual = (props.dm[i] * ∇ₐ_00 + props.dm[i + 1] * ∇ₐ_p1)/(props.dm[i] + props.dm[i + 1])
        update_face_dual_data!(props.∇ₐ_face[i], ∇ₐ_face_dual)

        L₀_dual = get_face_00_dual(props.L[i])*LSUN
        ∇ᵣ_dual = 3κface_dual * L₀_dual * exp(lnP_face_dual) /
                    (16π * CRAD * CLIGHT * CGRAV * props.m[i] * exp(4*lnT_face_dual))
        update_face_dual_data!(props.∇ᵣ_face[i], ∇ᵣ_dual)
    end
end

"""
    function copy_mesh!(sm, props_in, props_out)

Copies over the mesh quantities from `props_in` into `props_out`
This is a substitute for when remeshing is disabled.
"""
function copy_mesh_properties!(sm, props_out, props_in)
    Threads.@threads for i = 1:(props_in.nz)
        props_out.m[i] = props_in.m[i]
        props_out.dm[i] = props_in.dm[i]
        for j = 1:(sm.nvars)
            props_out.ind_vars[(i - 1) * sm.nvars + j] = props_in.ind_vars[(i - 1) * sm.nvars + j]
        end
    end
end

function copy_scalar_properties!(props_out, props_in)
    props_out.time = props_in.time
    props_out.dt = props_in.dt
    props_out.model_number = props_in.model_number
    props_out.mstar = props_in.mstar
end