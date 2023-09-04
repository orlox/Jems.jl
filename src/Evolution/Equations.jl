"""
    equationHSE(sm::StellarModel, k::Int,
                varm1::Matrix{TT}, var00::Matrix{TT}, varp1::Matrix{TT},
                eosm1::EOSResults{TT}, eos00::EOSResults{TT}, eosp1::EOSResults{TT},
                κm1::TT, κ00::TT, κp1::TT)::TT where {TT<:Real}

Default equation of hydrostatic equilibrium. Evaluates for cell `k` of StellarModel `sm` to what degree hydrostatic
equilibrium is satisfied.

# Arguments

  - `sm`: Stellar Model
  - `k`: cell number to consider
  - `varm1`: Matrix holding the dual numbers of the previous cell (`k-1`)
  - `var00`: Matrix holding the dual numbers of this cell (`k`)
  - `varp1`: Matrix holding the dual numbers of the next cell (`k+1`)
  - `eosm1`: EOSResults object holding the results of the EOS evaluation of the previous cell (`k-1`)
  - `eos00`: EOSResults object holding the results of the EOS evaluation of the current cell (`k`)
  - `eosp1`: EOSResults object holding the results of the EOS evaluation of the next cell (`k+1`)
  - `κm1`: Opacity evaluated at the previous cell (`k-1`)
  - `κ00`: Opacity evaluated at the current cell (`k`)
  - `κp1`: Opacity evaluated at the next cell (`k+1`)

# Returns

Residual of comparing dlnP/dm with -GM/4πr^4, where the latter is evaluated at the face of cell `k` and `k+1`.
"""
function equationHSE(sm::StellarModel, k::Int,
                     varm1::Matrix{TT}, var00::Matrix{TT}, varp1::Matrix{TT},
                     eosm1::EOSResults{TT}, eos00::EOSResults{TT}, eosp1::EOSResults{TT},
                     κm1::TT, κ00::TT, κp1::TT) where {TT<:Real}
    if k == sm.nz  # atmosphere boundary condition
        lnP₀ = var00[k, sm.vari[:lnP]]
        r₀ = exp(var00[k, sm.vari[:lnr]])
        g₀ = CGRAV * sm.mstar / r₀^2
        return lnP₀ - log(2g₀ / (3κ00))  # Eddington gray, ignoring radiation pressure term
    end
    lnP₊ = varp1[k, sm.vari[:lnP]]
    lnP₀ = var00[k, sm.vari[:lnP]]
    lnPface = (sm.dm[k] * lnP₀ + sm.dm[k + 1] * lnP₊) / (sm.dm[k] + sm.dm[k + 1])
    r₀ = exp(var00[k, sm.vari[:lnr]])
    dm = (sm.m[k + 1] - sm.m[k])

    return (exp(lnPface) * (lnP₊ - lnP₀) / dm + CGRAV * sm.m[k] / (4π * r₀^4)) /
           (CGRAV * sm.m[k] / (4π * r₀^4))
end

"""
    equationT(sm::StellarModel, k::Int,
              varm1::Matrix{TT}, var00::Matrix{TT}, varp1::Matrix{TT},
              eosm1::EOSResults{TT}, eos00::EOSResults{TT}, eosp1::EOSResults{TT},
              κm1::TT, κ00::TT, κp1::TT)::TT where {TT<:Real}

Default equation of energy transport, evaluated for cell `k` of StellarModel `sm`.

# Arguments

Identical to [`equationHSE`](@ref) for compatibility with [`TypeStableEquation`](@ref)

# Returns

Residual of comparing dlnT/dm with -∇*GMT/4πr^4P, where the latter is evaluation at the face of cell `k` and `k+1`.
"""
function equationT(sm::StellarModel, k::Int,
                   varm1::Matrix{TT}, var00::Matrix{TT}, varp1::Matrix{TT},
                   eosm1::EOSResults{TT}, eos00::EOSResults{TT}, eosp1::EOSResults{TT},
                   κm1::TT, κ00::TT, κp1::TT)::TT where {TT<:Real}
    if k == sm.nz  # atmosphere boundary condition
        lnT₀ = var00[k, sm.vari[:lnT]]
        L₀ = var00[k, sm.vari[:lum]] * LSUN
        r₀ = exp(var00[k, sm.vari[:lnr]])
        return lnT₀ - log(L₀ / (BOLTZ_SIGMA * 4π * r₀^2)) / 4  # Eddington gray, ignoring radiation pressure term
    end
    κface = exp((sm.dm[k] * log(κ00) + sm.dm[k + 1] * log(κp1)) / (sm.dm[k] + sm.dm[k + 1]))
    L₀ = var00[k, sm.vari[:lum]] * LSUN
    r₀ = exp(var00[k, sm.vari[:lnr]])
    Pface = exp((sm.dm[k] * var00[k, sm.vari[:lnP]] + sm.dm[k + 1] * varp1[k, sm.vari[:lnP]]) /
                (sm.dm[k] + sm.dm[k + 1]))
    lnT₊ = varp1[k, sm.vari[:lnT]]
    lnT₀ = var00[k, sm.vari[:lnT]]
    Tface = exp((sm.dm[k] * lnT₀ + sm.dm[k + 1] * lnT₊) / (sm.dm[k] + sm.dm[k + 1]))

    ∇ᵣ = 3κface * L₀ * Pface / (16π * CRAD * CLIGHT * CGRAV * sm.m[k] * Tface^4)
    ∇ₐ = (sm.dm[k] * eos00.∇ₐ + sm.dm[k + 1] * eosp1.∇ₐ) / (sm.dm[k] + sm.dm[k + 1])

    if (∇ᵣ < ∇ₐ)
        return (Tface * (lnT₊ - lnT₀) / sm.dm[k] +
                CGRAV * sm.m[k] * Tface / (4π * r₀^4 * Pface) * ∇ᵣ) /
               (CGRAV * sm.m[k] * Tface / (4π * r₀^4 * Pface))  # only radiative transport
    else  # should do convection here
        return (Tface * (lnT₊ - lnT₀) / sm.dm[k] +
                CGRAV * sm.m[k] * Tface / (4π * r₀^4 * Pface) * ∇ₐ) /
               (CGRAV * sm.m[k] * Tface / (4π * r₀^4 * Pface))  # only radiative transport
    end
end

"""
    equationLuminosity(sm::StellarModel, k::Int,
                       varm1::Matrix{TT}, var00::Matrix{TT}, varp1::Matrix{TT},
                       eosm1::EOSResults{TT}, eos00::EOSResults{TT}, eosp1::EOSResults{TT},
                       κm1::TT, κ00::TT, κp1::TT)::TT where {TT<:Real}

Default equation of energy generation, evaluated for cell `k` of StellarModel `sm`.

# Arguments

Identical to [`equationHSE`](@ref) for compatibility with [`TypeStableEquation`](@ref)

# Returns

Residual of comparing dL/dm with ϵnuc - cₚ * dT/dt - (δ / ρ) * dP/dt
"""
function equationLuminosity(sm::StellarModel, k::Int,
                            varm1::Matrix{TT}, var00::Matrix{TT}, varp1::Matrix{TT},
                            eosm1::EOSResults{TT}, eos00::EOSResults{TT}, eosp1::EOSResults{TT},
                            κm1::TT, κ00::TT, κp1::TT)::TT where {TT<:Real}
    if k > 1
        L₋ = varm1[k, sm.vari[:lum]] * LSUN  # change it if not at first cell
    else
        L₋::TT = 0  # central luminosity is zero at first cell
    end
    L₀ = var00[k, sm.vari[:lum]] * LSUN
    ρ₀ = eos00.ρ
    cₚ = eos00.cₚ
    δ = eos00.δ
    dTdt = (exp(var00[k, sm.vari[:lnT]]) - exp(sm.ssi.lnT[k])) / sm.ssi.dt
    dPdt = (exp(var00[k, sm.vari[:lnP]]) - exp(sm.ssi.lnP[k])) / sm.ssi.dt

    ϵnuc = 0.1 * var00[k, sm.vari[:H1]]^2 * ρ₀ * (exp(var00[k, sm.vari[:lnT]]) / 1e6)^4 +
           0.1 * var00[k, sm.vari[:H1]] * ρ₀ * (exp(var00[k, sm.vari[:lnT]]) / 1e7)^18
    return ((L₀ - L₋) / sm.dm[k] - ϵnuc + cₚ * dTdt - (δ / ρ₀) * dPdt)  # no neutrinos
end
"""
    equationContinuity(sm::StellarModel, k::Int,
                       varm1::Matrix{TT}, var00::Matrix{TT}, varp1::Matrix{TT},
                       eosm1::EOSResults{TT}, eos00::EOSResults{TT}, eosp1::EOSResults{TT},
                       κm1::TT, κ00::TT, κp1::TT)::TT where {TT<:Real}

Default equation of mass continuity, evaluated for cell `k` of StellarModel `sm`.

# Arguments

Identical to [`equationHSE`](@ref) for compatibility with [`TypeStableEquation`](@ref)

# Returns

Residual of comparing dr^3/dm with 3/(4πρ)
"""
function equationContinuity(sm::StellarModel, k::Int,
                            varm1::Matrix{TT}, var00::Matrix{TT}, varp1::Matrix{TT},
                            eosm1::EOSResults{TT}, eos00::EOSResults{TT}, eosp1::EOSResults{TT},
                            κm1::TT, κ00::TT, κp1::TT)::TT where {TT<:Real}
    ρ₀ = eos00.ρ
    r₀ = exp(var00[k, sm.vari[:lnr]])
    if k > 1  # get inner radius
        r₋ = exp(varm1[k, sm.vari[:lnr]])
    else
        r₋::TT = 0  # central radius is zero at first cell
    end

    # expected_r₀ = r₋ + dm/(4π*r₋^2*ρ)
    expected_dr³_dm = 3 / (4π * ρ₀)
    actual_dr³_dm = (r₀^3 - r₋^3) / sm.dm[k]

    return (expected_dr³_dm - actual_dr³_dm) * ρ₀  # times ρ to make eq. dim-less
end

"""
    equationH1(sm::StellarModel, k::Int,
               varm1::Matrix{TT}, var00::Matrix{TT}, varp1::Matrix{TT},
               eosm1::EOSResults{TT}, eos00::EOSResults{TT}, eosp1::EOSResults{TT},
               κm1::TT, κ00::TT, κp1::TT)::TT where {TT<:Real}

Default equation of Hydrogen 1 evoluation, evaluated for cell `k` of StellarModel `sm`.

# Arguments

Identical to [`equationHSE`](@ref) for compatibility with [`TypeStableEquation`](@ref)

# Returns

Residual of comparing dX_H1/dt with its computed reaction rate
"""
function equationH1(sm::StellarModel, k::Int,
                    varm1::Matrix{TT}, var00::Matrix{TT}, varp1::Matrix{TT},
                    eosm1::EOSResults{TT}, eos00::EOSResults{TT}, eosp1::EOSResults{TT},
                    κm1::TT, κ00::TT, κp1::TT)::TT where {TT<:Real}
    ρ₀ = eos00.ρ
    ϵnuc = 0.1 * var00[k, sm.vari[:H1]]^2 * ρ₀ * (exp(var00[k, sm.vari[:lnT]]) / 1e6)^4 +
           0.1 * var00[k, sm.vari[:H1]] * ρ₀ * (exp(var00[k, sm.vari[:lnT]]) / 1e7)^18
    rate_per_unit_mass = 4 * ϵnuc / ((4 * Chem.isotope_list[:H1].mass - Chem.isotope_list[:He4].mass) * AMU * CLIGHT^2)

    Xi = sm.ssi.ind_vars[(k - 1) * sm.nvars + sm.vari[:H1]]

    return (var00[k, sm.vari[:H1]] - Xi) / sm.ssi.dt +
           Chem.isotope_list[:H1].mass * AMU * rate_per_unit_mass
end

"""
    equationHe4(sm::StellarModel, k::Int,
               varm1::Matrix{TT}, var00::Matrix{TT}, varp1::Matrix{TT},
               eosm1::EOSResults{TT}, eos00::EOSResults{TT}, eosp1::EOSResults{TT},
               κm1::TT, κ00::TT, κp1::TT)::TT where {TT<:Real}

Default equation of Helium 4 evoluation, evaluated for cell `k` of StellarModel `sm`.

# Arguments

Identical to [`equationHSE`](@ref) for compatibility with [`TypeStableEquation`](@ref)

# Returns

Residual of comparing He4 content with 1 minus the H1 content.
"""
function equationHe4(sm::StellarModel, k::Int,
                     varm1::Matrix{TT}, var00::Matrix{TT}, varp1::Matrix{TT},
                     eosm1::EOSResults{TT}, eos00::EOSResults{TT}, eosp1::EOSResults{TT},
                     κm1::TT, κ00::TT, κp1::TT)::TT where {TT<:Real}
    return var00[k, sm.vari[:He4]] + var00[k, sm.vari[:H1]] - 1.0
end
