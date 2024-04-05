#####
##### Precipitation models
#####

import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.MicrophysicsFlexible as CMF
import CloudMicrophysics as CM
import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Operators as Operators
import ClimaCore.Fields as Fields
import ClimaCore.Utilities: half

precipitation_cache(Y, atmos::AtmosModel) =
    precipitation_cache(Y, atmos.precip_model)

#####
##### No Precipitation
#####

precipitation_cache(Y, precip_model::NoPrecipitation) = (;)
precipitation_tendency!(Yₜ, Y, p, t, colidx, ::NoPrecipitation) = nothing

#####
##### 0-Moment without sgs scheme or with diagnostic/prognostic edmf
#####

function precipitation_cache(Y, precip_model::Microphysics0Moment)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        ᶜS_ρq_tot = similar(Y.c, FT),
        ᶜ3d_rain = similar(Y.c, FT),
        ᶜ3d_snow = similar(Y.c, FT),
        col_integrated_rain = zeros(
            axes(Fields.level(Geometry.WVector.(Y.f.u₃), half)),
        ),
        col_integrated_snow = zeros(
            axes(Fields.level(Geometry.WVector.(Y.f.u₃), half)),
        ),
    )
end

function compute_precipitation_cache!(Y, p, colidx, ::Microphysics0Moment, _)
    (; params, dt) = p
    (; ᶜts) = p.precomputed
    (; ᶜS_ρq_tot) = p.precipitation
    cm_params = CAP.microphysics_params(params)
    thermo_params = CAP.thermodynamics_params(params)
    @. ᶜS_ρq_tot[colidx] =
        Y.c.ρ[colidx] * q_tot_precipitation_sources(
            Microphysics0Moment(),
            thermo_params,
            cm_params,
            dt,
            Y.c.ρq_tot[colidx] / Y.c.ρ[colidx],
            ᶜts[colidx],
        )
end

function compute_precipitation_cache!(
    Y,
    p,
    colidx,
    ::Microphysics0Moment,
    ::DiagnosticEDMFX,
)
    # For environment we multiply by grid mean ρ and not byᶜρa⁰
    # I.e. assuming a⁰=1

    (; ᶜS_q_tot⁰, ᶜS_q_totʲs, ᶜρaʲs) = p.precomputed
    (; ᶜS_ρq_tot) = p.precipitation
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    ρ = Y.c.ρ

    @. ᶜS_ρq_tot[colidx] = ᶜS_q_tot⁰[colidx] * ρ[colidx]
    for j in 1:n
        @. ᶜS_ρq_tot[colidx] += ᶜS_q_totʲs.:($$j)[colidx] * ᶜρaʲs.:($$j)[colidx]
    end
end

function compute_precipitation_cache!(
    Y,
    p,
    colidx,
    ::Microphysics0Moment,
    ::PrognosticEDMFX,
)
    (; ᶜS_q_tot⁰, ᶜS_q_totʲs, ᶜρa⁰) = p.precomputed
    (; ᶜS_ρq_tot) = p.precipitation
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)

    @. ᶜS_ρq_tot[colidx] = ᶜS_q_tot⁰[colidx] * ᶜρa⁰[colidx]
    for j in 1:n
        @. ᶜS_ρq_tot[colidx] +=
            ᶜS_q_totʲs.:($$j)[colidx] * Y.c.sgsʲs.:($$j).ρa[colidx]
    end
end

function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    precip_model::Microphysics0Moment,
)
    (; ᶜT, ᶜΦ) = p.core
    (; ᶜts,) = p.precomputed  # assume ᶜts has been updated
    (; params) = p
    (; turbconv_model) = p.atmos
    (;
        ᶜ3d_rain,
        ᶜ3d_snow,
        ᶜS_ρq_tot,
        col_integrated_rain,
        col_integrated_snow,
    ) = p.precipitation
    (; col_integrated_precip_energy_tendency,) = p.conservation_check

    thermo_params = CAP.thermodynamics_params(params)
    compute_precipitation_cache!(Y, p, colidx, precip_model, turbconv_model)
    if !isnothing(Yₜ)
        @. Yₜ.c.ρq_tot[colidx] += ᶜS_ρq_tot[colidx]
        @. Yₜ.c.ρ[colidx] += ᶜS_ρq_tot[colidx]
    end
    T_freeze = TD.Parameters.T_freeze(thermo_params)

    # update precip in cache for coupler's use
    # 3d rain and snow
    @. ᶜT[colidx] = TD.air_temperature(thermo_params, ᶜts[colidx])
    @. ᶜ3d_rain[colidx] = ifelse(ᶜT[colidx] >= T_freeze, ᶜS_ρq_tot[colidx], 0)
    @. ᶜ3d_snow[colidx] = ifelse(ᶜT[colidx] < T_freeze, ᶜS_ρq_tot[colidx], 0)
    Operators.column_integral_definite!(
        col_integrated_rain[colidx],
        ᶜ3d_rain[colidx],
    )
    Operators.column_integral_definite!(
        col_integrated_snow[colidx],
        ᶜ3d_snow[colidx],
    )

    if :ρe_tot in propertynames(Y.c)
        #TODO - this is a hack right now. But it will be easier to clean up
        # once we drop the support for the old EDMF code
        if turbconv_model isa DiagnosticEDMFX && !isnothing(Yₜ)
            @. Yₜ.c.ρe_tot[colidx] +=
                sum(
                    p.precomputed.ᶜS_q_totʲs[colidx] *
                    p.precomputed.ᶜρaʲs[colidx] *
                    p.precomputed.ᶜS_e_totʲs_helper[colidx],
                ) +
                p.precomputed.ᶜS_q_tot⁰[colidx] *
                Y.c.ρ[colidx] *
                e_tot_0M_precipitation_sources_helper(
                    thermo_params,
                    ᶜts[colidx],
                    ᶜΦ[colidx],
                )
        elseif !isnothing(Yₜ)
            ρe_tot_tend_colidx = p.scratch.ᶜtemp_scalar_3[colidx]
            @. ρe_tot_tend_colidx =
                ᶜS_ρq_tot[colidx] * e_tot_0M_precipitation_sources_helper(
                    thermo_params,
                    ᶜts[colidx],
                    ᶜΦ[colidx],
                )

            @. Yₜ.c.ρe_tot[colidx] += ρe_tot_tend_colidx
            Operators.column_integral_definite!(
                col_integrated_precip_energy_tendency[colidx],
                ρe_tot_tend_colidx,
            )
        end
    end
    return nothing
end

#####
##### 1-Moment without sgs scheme
#####

function precipitation_cache(Y, precip_model::Microphysics1Moment)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        ᶜSqₜᵖ = similar(Y.c, FT),
        ᶜSqᵣᵖ = similar(Y.c, FT),
        ᶜSqₛᵖ = similar(Y.c, FT),
        ᶜSeₜᵖ = similar(Y.c, FT),
    )
end

function compute_precipitation_cache!(Y, p, colidx, ::Microphysics1Moment, _)
    FT = Spaces.undertype(axes(Y.c))
    (; dt) = p
    (; ᶜts) = p.precomputed
    (; ᶜΦ) = p.core
    (; ᶜSqₜᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ, ᶜSeₜᵖ) = p.precipitation
    ᶜSᵖ = p.scratch.ᶜtemp_scalar
    ᶜSᵖ_snow = p.scratch.ᶜtemp_scalar_2

    # get thermodynamics and 1-moment microphysics params
    (; params) = p
    cmp = CAP.microphysics_params(params)
    thp = CAP.thermodynamics_params(params)

    # zero out the helper source terms
    @. ᶜSqₜᵖ[colidx] = FT(0)
    @. ᶜSqᵣᵖ[colidx] = FT(0)
    @. ᶜSqₛᵖ[colidx] = FT(0)
    @. ᶜSeₜᵖ[colidx] = FT(0)

    # compute precipitation source terms
    # TODO - need version of this for EDMF SGS
    compute_precipitation_sources!(
        ᶜSᵖ[colidx],
        ᶜSᵖ_snow[colidx],
        ᶜSqₜᵖ[colidx],
        ᶜSqᵣᵖ[colidx],
        ᶜSqₛᵖ[colidx],
        ᶜSeₜᵖ[colidx],
        Y.c.ρ[colidx],
        Y.c.ρq_rai[colidx],
        Y.c.ρq_sno[colidx],
        ᶜts[colidx],
        ᶜΦ[colidx],
        dt,
        cmp,
        thp,
    )

    # compute precipitation sinks
    # (For now only done on the grid mean)
    compute_precipitation_sinks!(
        ᶜSᵖ[colidx],
        ᶜSqₜᵖ[colidx],
        ᶜSqᵣᵖ[colidx],
        ᶜSqₛᵖ[colidx],
        ᶜSeₜᵖ[colidx],
        Y.c.ρ[colidx],
        Y.c.ρq_rai[colidx],
        Y.c.ρq_sno[colidx],
        ᶜts[colidx],
        ᶜΦ[colidx],
        dt,
        cmp,
        thp,
    )
end

function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    precip_model::Microphysics1Moment,
)
    (; turbconv_model) = p.atmos
    compute_precipitation_cache!(Y, p, colidx, precip_model, turbconv_model)

    (; ᶜSqₜᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ, ᶜSeₜᵖ) = p.precipitation

    # Update grid mean tendencies
    @. Yₜ.c.ρ[colidx] += Y.c.ρ[colidx] * ᶜSqₜᵖ[colidx]
    @. Yₜ.c.ρq_tot[colidx] += Y.c.ρ[colidx] * ᶜSqₜᵖ[colidx]
    @. Yₜ.c.ρe_tot[colidx] += Y.c.ρ[colidx] * ᶜSeₜᵖ[colidx]
    @. Yₜ.c.ρq_rai[colidx] += Y.c.ρ[colidx] * ᶜSqᵣᵖ[colidx]
    @. Yₜ.c.ρq_sno[colidx] += Y.c.ρ[colidx] * ᶜSqₛᵖ[colidx]

    return nothing
end

#####
##### TODO N-Moment without sgs scheme
#####

function precipitation_cache(Y, precip_model::MicrophysicsNMoment)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        ᶜSqₜᵖ = similar(Y.c, FT),
        Smc0 = similar(Y.c, FT),
        Smc1 = similar(Y.c, FT),
        Smr0 = similar(Y.c, FT),
        Smr1 = similar(Y.c, FT),
        Smr2 = similar(Y.c, FT),
        ᶜSeₜᵖ = similar(Y.c, FT),
        cloudy_cls = similar(Y.c, CMF.CLSetup{FT})
    )
end

function compute_precipitation_cache!(Y, p, colidx, ::MicrophysicsNMoment, _)
    FT = Spaces.undertype(axes(Y.c))
    (; dt) = p
    (; ᶜts) = p.precomputed
    (; ᶜΦ) = p.core
    (; ᶜSqₜᵖ, Smc0, Smc1, Smr0, Smr1, Smr2, ᶜSeₜᵖ, cloudy_info) = p.precipitation
    ᶜSᵖ = p.scratch.temp_data

    # get thermodynamics and N-moment microphysics params
    (; params) = p
    cmp = CAP.microphysics_params(params)
    thp = CAP.thermodynamics_params(params)

    # zero out the helper source terms
    @. ᶜSqₜᵖ[colidx] = FT(0)
    @. Smc0[colidx] = FT(0)
    @. Smc1[colidx] = FT(0)
    @. Smr0[colidx] = FT(0)
    @. Smr1[colidx] = FT(0)
    @. Smr2[colidx] = FT(0)
    @. ᶜSeₜᵖ[colidx] = FT(0)

    # compute precipitation source terms
    compute_Nmoment_tendencies!(
        ᶜSᵖ[colidx],
        Smc0[colidx],
        Smc1[colidx],
        Smr0[colidx],
        Smr1[colidx],
        Smr2[colidx],
        ᶜSeₜᵖ[colidx],
        cloudy_info[colidx],
        Y.c.ρ[colidx],
        Y.c.mom0_clo[colidx],
        Y.c.mom1_clo[colidx],
        Y.c.mom0_rai[colidx],
        Y.c.mom1_rai[colidx],
        Y.c.mom2_rai[colidx],
        ᶜts[colidx],
        ᶜΦ[colidx],
        dt,
        cmp,
        thp,
    )
end

function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    precip_model::MicrophysicsNMoment,
)
    (; turbconv_model) = p.atmos
    compute_precipitation_cache!(Y, p, colidx, precip_model, turbconv_model)

    (; ᶜSqₜᵖ, Smc0, Smc1, Smr0, Smr1, Smr2, ᶜSeₜᵖ) = p.precipitation

    # Update grid mean tendencies
    @. Yₜ.c.ρ[colidx] += Y.c.ρ[colidx] * ᶜSqₜᵖ[colidx]
    @. Yₜ.c.ρq_tot[colidx] += Y.c.ρ[colidx] * ᶜSqₜᵖ[colidx]
    @. Yₜ.c.ρe_tot[colidx] += Y.c.ρ[colidx] * ᶜSeₜᵖ[colidx]
    @. Yₜ.c.mom0_clo[colidx] += Smc0[colidx]
    @. Yₜ.c.mom1_clo[colidx] += Smc1[colidx]
    @. Yₜ.c.mom0_rai[colidx] += Smr0[colidx]
    @. Yₜ.c.mom1_rai[colidx] += Smr1[colidx]
    @. Yₜ.c.mom2_rai[colidx] += Smr2[colidx]

    return nothing
end

