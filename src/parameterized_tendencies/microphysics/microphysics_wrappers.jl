# A set of wrappers for using CloudMicrophysics.jl functions inside EDMFX loops

import Thermodynamics as TD
import CloudMicrophysics.Microphysics0M as CM0
import CloudMicrophysics.MicrophysicsFlexible as CMF

# define some aliases and functions to make the code more readable
const Iₗ = TD.internal_energy_liquid
const Iᵢ = TD.internal_energy_ice
const Lf = TD.latent_heat_fusion
const Tₐ = TD.air_temperature
const PP = TD.PhasePartition
const qᵥ = TD.vapor_specific_humidity
qₗ(thp, ts) = TD.PhasePartition(thp, ts).liq
qᵢ(thp, ts) = TD.PhasePartition(thp, ts).ice

# helper function to safely get precipitation from state
function qₚ(ρqₚ::FT, ρ::FT) where {FT}
    return max(FT(0), ρqₚ / ρ)
end

# helper function to limit the tendency
function limit(q::FT, dt::FT) where {FT}
    return q / dt / 5
end

"""
    q_tot_precipitation_sources(precip_model, thp, cmp, dt, qₜ, ts)

 - precip_model - a type for precipitation scheme choice
 - thp, cmp - structs with thermodynamic and microphysics parameters
 - dt - model time step
 - qₜ - total water specific humidity
 - ts - thermodynamic state (see Thermodynamics.jl package for details)

Returns the qₜ source term due to precipitation formation
defined as Δm_tot / (m_dry + m_tot)
"""
function q_tot_precipitation_sources(
    ::NoPrecipitation,
    thp,
    cmp,
    dt,
    qₜ::FT,
    ts,
) where {FT <: Real}
    return FT(0)
end
function q_tot_precipitation_sources(
    ::Microphysics0Moment,
    thp,
    cmp,
    dt,
    qₜ::FT,
    ts,
) where {FT <: Real}
    return -min(max(qₜ, 0) / dt, -CM0.remove_precipitation(cmp, PP(thp, ts)))
end

"""
    e_tot_0M_precipitation_sources_helper(thp, ts, Φ)

 - thp - set with thermodynamics parameters
 - ts - thermodynamic state (see td package for details)
 - Φ - geopotential

Returns the total energy source term multiplier from precipitation formation
for the 0-moment scheme
"""
function e_tot_0M_precipitation_sources_helper(
    thp,
    ts,
    Φ::FT,
) where {FT <: Real}

    λ::FT = TD.liquid_fraction(thp, ts)

    return λ * Iₗ(thp, ts) + (1 - λ) * Iᵢ(thp, ts) + Φ
end

"""
    compute_precipitation_sources!(Sᵖ, Sᵖ_snow, Sqₜᵖ, Sqᵣᵖ, Sqₛᵖ, Seₜᵖ, ρ, ρqᵣ, ρqₛ, ts, Φ, dt, mp, thp)

 - Sᵖ, Sᵖ_snow - temporary containters to help compute precipitation source terms
 - Sqₜᵖ, Sqᵣᵖ, Sqₛᵖ, Seₜᵖ - cached storage for precipitation source terms
 - ρ - air density
 - ρqᵣ, ρqₛ - precipitation (rain and snow) densities
 - ts - thermodynamic state (see td package for details)
 - Φ - geopotential
 - dt - model time step
 - thp, cmp - structs with thermodynamic and microphysics parameters

Returns the q source terms due to precipitation formation from the 1-moment scheme.
The specific humidity source terms are defined as defined as Δmᵢ / (m_dry + m_tot)
where i stands for total, rain or snow.
Also returns the total energy source term due to the microphysics processes.
"""
function compute_precipitation_sources!(
    Sᵖ,
    Sᵖ_snow,
    Sqₜᵖ,
    Sqᵣᵖ,
    Sqₛᵖ,
    Seₜᵖ,
    ρ,
    ρqᵣ,
    ρqₛ,
    ts,
    Φ,
    dt,
    mp,
    thp,
)
    FT = eltype(Sqₜᵖ)
    #! format: off
    # rain autoconversion: q_liq -> q_rain
    @. Sᵖ = min(
        limit(qₗ(thp, ts), dt),
        CM1.conv_q_liq_to_q_rai(mp.pr.acnv1M, qₗ(thp, ts), true),
    )
    @. Sqₜᵖ -= Sᵖ
    @. Sqᵣᵖ += Sᵖ
    @. Seₜᵖ -= Sᵖ * (Iₗ(thp, ts) + Φ)

    # snow autoconversion assuming no supersaturation: q_ice -> q_snow
    @. Sᵖ = min(
        limit(qᵢ(thp, ts), dt),
        CM1.conv_q_ice_to_q_sno_no_supersat(mp.ps.acnv1M, qᵢ(thp, ts), true),
    )
    @. Sqₜᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ
    @. Seₜᵖ -= Sᵖ * (Iᵢ(thp, ts) + Φ)

    # accretion: q_liq + q_rain -> q_rain
    @. Sᵖ = min(
        limit(qₗ(thp, ts), dt),
        CM1.accretion(mp.cl, mp.pr, mp.tv.rain, mp.ce, qₗ(thp, ts), qₚ(ρqᵣ, ρ), ρ),
    )
    @. Sqₜᵖ -= Sᵖ
    @. Sqᵣᵖ += Sᵖ
    @. Seₜᵖ -= Sᵖ * (Iₗ(thp, ts) + Φ)

    # accretion: q_ice + q_snow -> q_snow
    @. Sᵖ = min(
        limit(qᵢ(thp, ts), dt),
        CM1.accretion(mp.ci, mp.ps, mp.tv.snow, mp.ce, qᵢ(thp, ts), qₚ(ρqₛ, ρ), ρ),
    )
    @. Sqₜᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ
    @. Seₜᵖ -= Sᵖ * (Iᵢ(thp, ts) + Φ)

    # accretion: q_liq + q_sno -> q_sno or q_rai
    # sink of cloud water via accretion cloud water + snow
    @. Sᵖ = min(
        limit(qₗ(thp, ts), dt),
        CM1.accretion(mp.cl, mp.ps, mp.tv.snow, mp.ce, qₗ(thp, ts), qₚ(ρqₛ, ρ), ρ),
    )
    # if T < T_freeze cloud droplets freeze to become snow
    # else the snow melts and both cloud water and snow become rain
    α(thp, ts) = TD.Parameters.cv_l(thp) / Lf(thp, ts) * (Tₐ(thp, ts) - mp.ps.T_freeze)
    @. Sᵖ_snow = ifelse(
        Tₐ(thp, ts) < mp.ps.T_freeze,
        Sᵖ,
        FT(-1) * min(Sᵖ * α(thp, ts), limit(qₚ(ρqₛ, ρ), dt)),
    )
    @. Sqₛᵖ += Sᵖ_snow
    @. Sqₜᵖ -= Sᵖ
    @. Sqᵣᵖ += ifelse(Tₐ(thp, ts) < mp.ps.T_freeze, FT(0), Sᵖ - Sᵖ_snow)
    @. Seₜᵖ -= ifelse(
        Tₐ(thp, ts) < mp.ps.T_freeze,
        Sᵖ * (Iᵢ(thp, ts) + Φ),
        Sᵖ * (Iₗ(thp, ts) + Φ) - Sᵖ_snow * (Iₗ(thp, ts) - Iᵢ(thp, ts)),
    )

    # accretion: q_ice + q_rai -> q_sno
    @. Sᵖ = min(
        limit(qᵢ(thp, ts), dt),
        CM1.accretion(mp.ci, mp.pr, mp.tv.rain, mp.ce, qᵢ(thp, ts), qₚ(ρqᵣ, ρ), ρ),
    )
    @. Sqₜᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ
    @. Seₜᵖ -= Sᵖ * (Iᵢ(thp, ts) + Φ)
    # sink of rain via accretion cloud ice - rain
    @. Sᵖ = min(
        limit(qₚ(ρqᵣ, ρ), dt),
        CM1.accretion_rain_sink(mp.pr, mp.ci, mp.tv.rain, mp.ce, qᵢ(thp, ts), qₚ(ρqᵣ, ρ), ρ),
    )
    @. Sqᵣᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ
    @. Seₜᵖ += Sᵖ * Lf(thp, ts)

    # accretion: q_rai + q_sno -> q_rai or q_sno
    @. Sᵖ = ifelse(
        Tₐ(thp, ts) < mp.ps.T_freeze,
        min(
            limit(qₚ(ρqᵣ, ρ), dt),
            CM1.accretion_snow_rain(mp.ps, mp.pr, mp.tv.rain, mp.tv.snow, mp.ce, qₚ(ρqₛ, ρ), qₚ(ρqᵣ, ρ), ρ),
        ),
        -min(
            limit(qₚ(ρqₛ, ρ), dt),
            CM1.accretion_snow_rain(mp.pr, mp.ps, mp.tv.snow, mp.tv.rain, mp.ce, qₚ(ρqᵣ, ρ), qₚ(ρqₛ, ρ), ρ),
        ),
    )
    @. Sqₛᵖ += Sᵖ
    @. Sqᵣᵖ -= Sᵖ
    @. Seₜᵖ += Sᵖ * Lf(thp, ts)
    #! format: on
end

"""
    compute_precipitation_sinks!(Sᵖ, Sqₜᵖ, Sqᵣᵖ, Sqₛᵖ, Seₜᵖ, ρ, ρqᵣ, ρqₛ, ts, Φ, dt, mp, thp)

 - Sᵖ - a temporary containter to help compute precipitation source terms
 - Sqₜᵖ, Sqᵣᵖ, Sqₛᵖ, Seₜᵖ - cached storage for precipitation source terms
 - ρ - air density
 - ρqᵣ, ρqₛ - precipitation (rain and snow) densities
 - ts - thermodynamic state (see td package for details)
 - Φ - geopotential
 - dt - model time step
 - thp, cmp - structs with thermodynamic and microphysics parameters

Returns the q source terms due to precipitation sinks from the 1-moment scheme.
The specific humidity source terms are defined as defined as Δmᵢ / (m_dry + m_tot)
where i stands for total, rain or snow.
Also returns the total energy source term due to the microphysics processes.
"""
function compute_precipitation_sinks!(
    Sᵖ,
    Sqₜᵖ,
    Sqᵣᵖ,
    Sqₛᵖ,
    Seₜᵖ,
    ρ,
    ρqᵣ,
    ρqₛ,
    ts,
    Φ,
    dt,
    mp,
    thp,
)
    FT = eltype(Sqₜᵖ)
    sps = (mp.ps, mp.tv.snow, mp.aps, thp)
    rps = (mp.pr, mp.tv.rain, mp.aps, thp)

    #! format: off
    # evaporation: q_rai -> q_vap
    @. Sᵖ = -min(
        limit(qₚ(ρqᵣ, ρ), dt),
        -CM1.evaporation_sublimation(rps..., PP(thp, ts), qₚ(ρqᵣ, ρ), ρ, Tₐ(thp, ts)),
    )
    @. Sqₜᵖ -= Sᵖ
    @. Sqᵣᵖ += Sᵖ
    @. Seₜᵖ -= Sᵖ * (Iₗ(thp, ts) + Φ)

    # melting: q_sno -> q_rai
    @. Sᵖ = min(
        limit(qₚ(ρqₛ, ρ), dt),
        CM1.snow_melt(sps..., qₚ(ρqₛ, ρ), ρ, Tₐ(thp, ts)),
    )
    @. Sqᵣᵖ += Sᵖ
    @. Sqₛᵖ -= Sᵖ
    @. Seₜᵖ -= Sᵖ * Lf(thp, ts)

    # deposition/sublimation: q_vap <-> q_sno
    @. Sᵖ = CM1.evaporation_sublimation(sps..., PP(thp, ts), qₚ(ρqₛ, ρ), ρ, Tₐ(thp, ts))
    @. Sᵖ = ifelse(
        Sᵖ > FT(0),
        min(limit(qᵥ(thp, ts), dt), Sᵖ),
        -min(limit(qₚ(ρqₛ, ρ), dt), FT(-1) * Sᵖ),
    )
    @. Sqₜᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ
    @. Seₜᵖ -= Sᵖ * (Iᵢ(thp, ts) + Φ)
    #! format: on
end

# TODO: Sources and sinks for NMoment