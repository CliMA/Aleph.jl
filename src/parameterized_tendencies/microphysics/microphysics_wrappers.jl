# A set of wrappers for using CloudMicrophysics.jl functions inside EDMFX loops

import Thermodynamics as TD
import CloudMicrophysics as CM
import CloudMicrophysics.Microphysics0M as CM0
import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.Microphysics2M as CM2
import CloudMicrophysics.MicrophysicsNonEq as CMNe
import CloudMicrophysics.Parameters as CMP
import Cloudy as CL

# define some aliases and functions to make the code more readable
const Iₗ = TD.internal_energy_liquid
const Iᵢ = TD.internal_energy_ice
const Lf = TD.latent_heat_fusion
const Tₐ = TD.air_temperature
const PP = TD.PhasePartition
const qᵥ = TD.vapor_specific_humidity
qₜ(thp, ts) = TD.PhasePartition(thp, ts).tot
qₗ(thp, ts) = TD.PhasePartition(thp, ts).liq
qᵢ(thp, ts) = TD.PhasePartition(thp, ts).ice
cᵥₗ(thp) = TD.Parameters.cv_l(thp)
cᵥᵢ(thp) = TD.Parameters.cv_i(thp)

# helper function to limit the tendency
function limit(q::FT, dt::FT, n::Int) where {FT}
    return q / dt / n
end

"""
    cloud_sources(cm_params, thp, ts, dt)

 - cm_params - CloudMicrophysics parameters struct for cloud water or ice condensate
 - thp - Thermodynamics parameters struct
 - ts - thermodynamics state
 - dt - model time step

Returns the condensation/evaporation or deposition/sublimation rate for
non-equilibrium cloud formation.
"""
function cloud_sources(cm_params::CMP.CloudLiquid{FT}, thp, ts, dt) where {FT}

    λ = TD.liquid_fraction(thp, Tₐ(thp, ts), TD.PhaseEquil)
    qᵥ_ex_liquid = λ * (qₜ(thp, ts) - TD.q_vap_saturation_liquid(thp, ts))

    # Ideally the logic whether to apply this source term, and what relaxation
    # timescale to use, should be based on the availability of CCNs and INPs.
    # We need a 2-moment microphysics scheme for that.

    # Additionally, to approximate the partitioning between the condensation/evaporation
    # and deposition/sublimation, we are scaling down the excess q by the liquid fraction.
    # This needs more thought.
    S = CMNe.conv_q_vap_to_q_liq_ice(
        cm_params,
        PP(FT(0), qᵥ_ex_liquid, FT(0)),
        PP(thp, ts),
    )
    return ifelse(
        S > FT(0),
        min(S, limit(qᵥ(thp, ts), dt, 2)),
        -min(abs(S), limit(qₗ(thp, ts), dt, 2)),
    )
end
function cloud_sources(cm_params::CMP.CloudIce{FT}, thp, ts, dt) where {FT}

    λ = TD.liquid_fraction(thp, Tₐ(thp, ts), TD.PhaseEquil)
    qᵥ_ex_ice = (1 - λ) * (qₜ(thp, ts) - TD.q_vap_saturation_ice(thp, ts))

    S = CMNe.conv_q_vap_to_q_liq_ice(
        cm_params,
        PP(FT(0), FT(0), qᵥ_ex_ice),
        PP(thp, ts),
    )
    return ifelse(
        S > FT(0),
        min(S, limit(qᵥ(thp, ts), dt, 2)),
        -min(abs(S), limit(qᵢ(thp, ts), dt, 2)),
    )
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
    compute_precipitation_sources!(Sᵖ, Sᵖ_snow, Sqₜᵖ, Sqᵣᵖ, Sqₛᵖ, Seₜᵖ, ρ, qᵣ, qₛ, ts, Φ, dt, mp, thp)

 - Sᵖ, Sᵖ_snow - temporary containters to help compute precipitation source terms
 - Sqₜᵖ, Sqᵣᵖ, Sqₛᵖ, Seₜᵖ - cached storage for precipitation source terms
 - ρ - air density
 - qᵣ, qₛ - precipitation (rain and snow) specific humidity
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
    qᵣ,
    qₛ,
    ts,
    Φ,
    dt,
    mp,
    thp,
)
    FT = eltype(thp)
    # @. Sqₜᵖ = FT(0) should work after fixing
    # https://github.com/CliMA/ClimaCore.jl/issues/1786
    @. Sqₜᵖ = ρ * FT(0)
    @. Sqᵣᵖ = ρ * FT(0)
    @. Sqₛᵖ = ρ * FT(0)
    @. Seₜᵖ = ρ * FT(0)

    #! format: off
    # rain autoconversion: q_liq -> q_rain
    @. Sᵖ = ifelse(
        mp.Ndp <= 0,
        CM1.conv_q_liq_to_q_rai(mp.pr.acnv1M, qₗ(thp, ts), true),
        CM2.conv_q_liq_to_q_rai(mp.var, qₗ(thp, ts), ρ, mp.Ndp),
    )
    @. Sᵖ = min(limit(qₗ(thp, ts), dt, 5), Sᵖ)
    @. Sqₜᵖ -= Sᵖ
    @. Sqᵣᵖ += Sᵖ
    @. Seₜᵖ -= Sᵖ * (Iₗ(thp, ts) + Φ)

    # snow autoconversion assuming no supersaturation: q_ice -> q_snow
    @. Sᵖ = min(
        limit(qᵢ(thp, ts), dt, 5),
        CM1.conv_q_ice_to_q_sno_no_supersat(mp.ps.acnv1M, qᵢ(thp, ts), true),
    )
    @. Sqₜᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ
    @. Seₜᵖ -= Sᵖ * (Iᵢ(thp, ts) + Φ)

    # accretion: q_liq + q_rain -> q_rain
    @. Sᵖ = min(
        limit(qₗ(thp, ts), dt, 5),
        CM1.accretion(mp.cl, mp.pr, mp.tv.rain, mp.ce, qₗ(thp, ts), qᵣ, ρ),
    )
    @. Sqₜᵖ -= Sᵖ
    @. Sqᵣᵖ += Sᵖ
    @. Seₜᵖ -= Sᵖ * (Iₗ(thp, ts) + Φ)

    # accretion: q_ice + q_snow -> q_snow
    @. Sᵖ = min(
        limit(qᵢ(thp, ts), dt, 5),
        CM1.accretion(mp.ci, mp.ps, mp.tv.snow, mp.ce, qᵢ(thp, ts), qₛ, ρ),
    )
    @. Sqₜᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ
    @. Seₜᵖ -= Sᵖ * (Iᵢ(thp, ts) + Φ)

    # accretion: q_liq + q_sno -> q_sno or q_rai
    # sink of cloud water via accretion cloud water + snow
    @. Sᵖ = min(
        limit(qₗ(thp, ts), dt, 5),
        CM1.accretion(mp.cl, mp.ps, mp.tv.snow, mp.ce, qₗ(thp, ts), qₛ, ρ),
    )
    # if T < T_freeze cloud droplets freeze to become snow
    # else the snow melts and both cloud water and snow become rain
    α(thp, ts) = cᵥₗ(thp) / Lf(thp, ts) * (Tₐ(thp, ts) - mp.ps.T_freeze)
    @. Sᵖ_snow = ifelse(
        Tₐ(thp, ts) < mp.ps.T_freeze,
        Sᵖ,
        FT(-1) * min(Sᵖ * α(thp, ts), limit(qₛ, dt, 5)),
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
        limit(qᵢ(thp, ts), dt, 5),
        CM1.accretion(mp.ci, mp.pr, mp.tv.rain, mp.ce, qᵢ(thp, ts), qᵣ, ρ),
    )
    @. Sqₜᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ
    @. Seₜᵖ -= Sᵖ * (Iᵢ(thp, ts) + Φ)
    # sink of rain via accretion cloud ice - rain
    @. Sᵖ = min(
        limit(qᵣ, dt, 5),
        CM1.accretion_rain_sink(mp.pr, mp.ci, mp.tv.rain, mp.ce, qᵢ(thp, ts), qᵣ, ρ),
    )
    @. Sqᵣᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ
    @. Seₜᵖ += Sᵖ * Lf(thp, ts)

    # accretion: q_rai + q_sno -> q_rai or q_sno
    @. Sᵖ = ifelse(
        Tₐ(thp, ts) < mp.ps.T_freeze,
        min(
            limit(qᵣ, dt, 5),
            CM1.accretion_snow_rain(mp.ps, mp.pr, mp.tv.rain, mp.tv.snow, mp.ce, qₛ, qᵣ, ρ),
        ),
        -min(
            limit(qₛ, dt, 5),
            CM1.accretion_snow_rain(mp.pr, mp.ps, mp.tv.snow, mp.tv.rain, mp.ce, qᵣ, qₛ, ρ),
        ),
    )
    @. Sqₛᵖ += Sᵖ
    @. Sqᵣᵖ -= Sᵖ
    @. Seₜᵖ += Sᵖ * Lf(thp, ts)
    #! format: on
end

"""
    compute_precipitation_heating(Seₜᵖ, ᶜwᵣ, ᶜwₛ, ᶜu, qᵣ, qₛ, ᶜts, thp)

 - Seₜᵖ - cached storage for precipitation energy source terms
 - ᶜwᵣ, ᶜwₛ - rain and snow terminal velocities
 - ᶜu - air velocity
 - qᵣ, qₛ - precipitation (rain and snow) specific humidities
 - ᶜts - thermodynamic state (see td package for details)
 - ᶜ∇T - cached temporary variable to store temperature gradient
 - thp - structs with thermodynamic and microphysics parameters

 Augments the energy source terms with heat exchange between air
 and precipitating species, following eq. 36 from Raymond 2013
 doi:10.1002/jame.20050 and assuming that precipitation has the same
 temperature as the surrounding air.
"""
function compute_precipitation_heating!(
    ᶜSeₜᵖ,
    ᶜwᵣ,
    ᶜwₛ,
    ᶜu,
    ᶜqᵣ,
    ᶜqₛ,
    ᶜts,
    ᶜ∇T,
    thp,
)
    # TODO - at some point we want to switch to assuming that precipitation
    # is at wet bulb temperature

    # compute full temperature gradient
    @. ᶜ∇T = CT123(ᶜgradᵥ(ᶠinterp(Tₐ(thp, ᶜts))))
    @. ᶜ∇T += CT123(gradₕ(Tₐ(thp, ᶜts)))
    # dot product with effective velocity of precipitation
    # (times q and specific heat)
    @. ᶜSeₜᵖ -= dot(ᶜ∇T, (ᶜu - C123(Geometry.WVector(ᶜwᵣ)))) * cᵥₗ(thp) * ᶜqᵣ
    @. ᶜSeₜᵖ -= dot(ᶜ∇T, (ᶜu - C123(Geometry.WVector(ᶜwₛ)))) * cᵥᵢ(thp) * ᶜqₛ
end
"""
    compute_precipitation_sinks!(Sᵖ, Sqₜᵖ, Sqᵣᵖ, Sqₛᵖ, Seₜᵖ, ρ, qᵣ, qₛ, ts, Φ, dt, mp, thp)

 - Sᵖ - a temporary containter to help compute precipitation source terms
 - Sqₜᵖ, Sqᵣᵖ, Sqₛᵖ, Seₜᵖ - cached storage for precipitation source terms
 - ρ - air density
 - qᵣ, qₛ - precipitation (rain and snow) specific humidities
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
    qᵣ,
    qₛ,
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
        limit(qᵣ, dt, 5),
        -CM1.evaporation_sublimation(rps..., PP(thp, ts), qᵣ, ρ, Tₐ(thp, ts)),
    )
    @. Sqₜᵖ -= Sᵖ
    @. Sqᵣᵖ += Sᵖ
    @. Seₜᵖ -= Sᵖ * (Iₗ(thp, ts) + Φ)

    # melting: q_sno -> q_rai
    @. Sᵖ = min(
        limit(qₛ, dt, 5),
        CM1.snow_melt(sps..., qₛ, ρ, Tₐ(thp, ts)),
    )
    @. Sqᵣᵖ += Sᵖ
    @. Sqₛᵖ -= Sᵖ
    @. Seₜᵖ -= Sᵖ * Lf(thp, ts)

    # deposition/sublimation: q_vap <-> q_sno
    @. Sᵖ = CM1.evaporation_sublimation(sps..., PP(thp, ts), qₛ, ρ, Tₐ(thp, ts))
    @. Sᵖ = ifelse(
        Sᵖ > FT(0),
        min(limit(qᵥ(thp, ts), dt, 5), Sᵖ),
        -min(limit(qₛ, dt, 5), FT(-1) * Sᵖ),
    )
    @. Sqₜᵖ -= Sᵖ
    @. Sqₛᵖ += Sᵖ
    @. Seₜᵖ -= Sᵖ * (Iᵢ(thp, ts) + Φ)
    #! format: on
end

"""
   get_weighted_vt(moments, pdists, cloudy_params)

 - moments - current distribution moments
 - pdists - current particledistributions
 - cloudy_params - parameters for Cloudy specific stuff

Returns the moment-weighted terminal velocities corresponding to the input momnents,
where the terminal velocity parameters are specific in cloudy_params
"""
function get_weighted_vt(FT, moments, pdists, cloudy_params)
    sed_flux = CL.Sedimentation.get_sedimentation_flux(pdists, cloudy_params.vel)
    ntuple(length(moments)) do i
        ifelse(moments[i] > FT(0), -1 * sed_flux[i] * cloudy_params.mom_norms[i] / moments[i], FT(0))
    end
end

"""
    update_pdists!(moments, pdists, cloudy_params)

 - moments - current distribution moments
 - old_pdists - previous particledistributions
 - cloudy_params - parameters for Cloudy specific stuff

Returns a new tuple of pdists with updated parameters based on the current moments
"""
function get_updated_pdists(moments, old_pdists, cloudy_params)
    mom_normed = moments ./ cloudy_params.mom_norms
    ntuple(length(cloudy_params.NProgMoms)) do i
        mom_i = mom_normed[CL.get_dist_moments_ind_range(cloudy_params.NProgMoms, i)] 
        if old_pdists[i] isa CL.ParticleDistributions.GammaPrimitiveParticleDistribution
            CL.ParticleDistributions.update_dist_from_moments(
                old_pdists[i],
                mom_i,
                param_range = (; :k => (1.0, 10.0)),
            )
        elseif old_pdists[i] isa CL.ParticleDistributions.LognormalPrimitiveParticleDistribution
            CL.ParticleDistributions.update_dist_from_moments(old_pdists[i], mom_i)
        else # Exponential or monodisperse
            CL.ParticleDistributions.update_dist_from_moments(old_pdists[i], mom_i)
        end
    end
end

"""
    separate_liq_rai(moments, pdists, cloudy_params)

 - moments - current distribution moments
 - pdists - current particledistributions
 - cloudy_params - parameters for Cloudy specific stuff

Returns a tuple of the same size as "moments", where the first two entries
are the liquid and rain mass density corresponding to the current distributions,
using the threshold size specific in cloudy_params
"""
function separate_liq_rai(FT, moments, pdists, cloudy_params)
    tmp = CL.ParticleDistributions.get_standard_N_q(pdists, cloudy_params.size_threshold / cloudy_params.norms[2])
    ntuple(length(moments)) do k
        if k == 1
            max(tmp.M_liq * cloudy_params.mom_norms[1], FT(0))
        elseif k == 2
            max(tmp.M_rai * cloudy_params.mom_norms[1], FT(0))
        else
            FT(0)
        end
    end
end


"""
   get_coal_sources(moments, pdists, cloudy_params, dt)

 - moments - current distribution moments
 - pdists - current particledistributions
 - cloudy_params - parameters for Cloudy specific stuff
 - dt - model time step, used for tendency limiters

Returns the moment tendencies coming from coalescence processes, using the 
coalescence dynamics info from cloudy_params
"""
function get_coal_sources(moments, pdists, cloudy_params, dt)
    dY_coal_tmp = CL.Coalescence.get_coal_ints(CL.EquationTypes.AnalyticalCoalStyle(), pdists, cloudy_params.coal_data)
    ntuple(length(moments)) do j
        ifelse(
            dY_coal_tmp[j] >= 0,
            dY_coal_tmp[j] * cloudy_params.mom_norms[j],
            max(dY_coal_tmp[j] * cloudy_params.mom_norms[j], -moments[j] / dt),
        )
    end
end


"""
   get_cond_evap_sources(FT, thp, mp, clp, ρq_tot, ρq_liq, moments, pdists, T, ρ, dt)

 - thermo_params
 - cmp - microphysics params, including the air properties
 - cloudy_params
 - ρq_tot 
 - ρq_liq
 - moments - current distribution moments
 - pdists - current particledistributions
 - ts - thermodynamic state
 - dt - model time step, used for tendency limiters

Returns the moment tendencies coming from coalescence processes, using the 
coalescence dynamics info from cloudy_params
"""
function get_cond_evap_sources(FT, thermo_params, cmp, cloudy_params, moments, pdists, ts, ρ, dt)
    q = TD.PhasePartition(thermo_params, ts)
    T = Tₐ(thermo_params, ts)
    ξ = CM.Common.G_func(cmp.aps, thermo_params, T, TD.Liquid())
    ξ_normed = ξ / FT(cloudy_params.norms[2]^(2 / 3))
    s = TD.supersaturation(thermo_params, q, ρ, T, TD.Liquid())
    dY_ce_tmp = CL.Condensation.get_cond_evap(pdists, s, ξ_normed) .* cloudy_params.mom_norms
    ntuple(length(moments)) do j
        if (j == cloudy_params.NProgMoms[1] + 1 && moments[j + 1] / moments[j] < 1e-11) ||
           (j == 1 && moments[j + 1] / moments[j] < 1e-14)
            -moments[j] / dt / 4
        else
            ifelse(dY_ce_tmp[j] >= 0, dY_ce_tmp[j], max(dY_ce_tmp[j], -moments[j] / dt))
        end
    end
end

"""
   get_M1_flux(moments, weighted_vt, cloudy_params)

 - moments - current distribution moments
 - weighted_vt - moment-weighted terminal velocities
 - cloudy_params

Returns the vertical flux of the first moment
"""
function get_M1_flux(moments, weighted_vt, cloudy_params)
    m1_ind = 2 - cloudy_params.NProgMoms[1]
    m1_fluxes = ntuple(length(cloudy_params.NProgMoms)) do k
        m1_ind = m1_ind + cloudy_params.NProgMoms[k]
        moments[m1_ind] * weighted_vt[m1_ind]
    end

    return sum(m1_fluxes)
end
