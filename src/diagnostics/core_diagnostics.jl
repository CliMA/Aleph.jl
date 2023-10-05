# This file is included in Diagnostics.jl
#
# README: Adding a new core diagnostic:
#
# In addition to the metadata (names, comments, ...), the most important step in adding a
# new DiagnosticVariable is defining its compute! function. `compute!` has to take four
# arguments: (out, state, cache, time), and as to write the diagnostic in place into the
# `out` variable.
#
# Often, it is possible to compute certain diagnostics only for specific models (e.g.,
# humidity for moist models). For that, it is convenient to adopt the following pattern:
#
# 1. Define a catch base function that does the computation we want to do for the case we know
# how to handle, for example
#
# function compute_hur!(
#     out,
#     state,
#     cache,
#     time,
#     ::T,
# ) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
#     thermo_params = CAP.thermodynamics_params(cache.params)
#     out .= TD.relative_humidity.(thermo_params, cache.ᶜts)
# end
#
# 2. Define a function that has the correct signature and calls this function
#
# compute_hur!(out, state, cache, time) =
#     compute_hur!(out, state, cache, time, cache.atmos.moisture_model)
#
# 3. Define a function that returns an error when the model is incorrect
#
# compute_hur!(_, _, _, _, model::T) where {T} =
#     error_diagnostic_variable("relative_humidity", model)
#
# We can also output a specific error message
#
# compute_hur!(_, _, _, _, model::T) where {T} =
#     error_diagnostic_variable("relative humidity makes sense only for moist models")

# General helper functions for undefined diagnostics for a particular model
error_diagnostic_variable(
    message = "Cannot compute $variable with model = $T",
) = error(message)
error_diagnostic_variable(variable, model::T) where {T} =
    error_diagnostic_variable("Cannot compute $variable with model = $T")

###
# Rho (3d)
###
add_diagnostic_variable!(
    short_name = "rhoa",
    long_name = "Air Density",
    standard_name = "air_density",
    units = "kg m^-3",
    comments = "Density of air",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(state.c.ρ)
        else
            out .= state.c.ρ
        end
    end,
)

###
# U velocity (3d)
###
add_diagnostic_variable!(
    short_name = "ua",
    long_name = "Eastward Wind",
    standard_name = "eastward_wind",
    units = "m s^-1",
    comments = "Eastward (zonal) wind component",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(Geometry.UVector.(cache.ᶜu).components.data.:1)
        else
            out .= Geometry.UVector.(cache.ᶜu).components.data.:1
        end
    end,
)

###
# V velocity (3d)
###
add_diagnostic_variable!(
    short_name = "va",
    long_name = "Northward Wind",
    standard_name = "northward_wind",
    units = "m s^-1",
    comments = "Northward (meridional) wind component",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(Geometry.VVector.(cache.ᶜu).components.data.:1)
        else
            out .= Geometry.VVector.(cache.ᶜu).components.data.:1
        end
    end,
)

###
# W velocity (3d)
###
# TODO: may want to convert to omega (Lagrangian pressure tendency) as standard output,
# but this is probably more useful for now
#
add_diagnostic_variable!(
    short_name = "wa",
    long_name = "Upward Air Velocity",
    standard_name = "upward_air_velocity",
    units = "m s^-1",
    comments = "Vertical wind component",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(Geometry.WVector.(cache.ᶜu).components.data.:1)
        else
            out .= Geometry.WVector.(cache.ᶜu).components.data.:1
        end
    end,
)

###
# Temperature (3d)
###
add_diagnostic_variable!(
    short_name = "ta",
    long_name = "Air Temperature",
    standard_name = "air_temperature",
    units = "K",
    comments = "Temperature of air",
    compute! = (out, state, cache, time) -> begin
        thermo_params = CAP.thermodynamics_params(cache.params)
        if isnothing(out)
            return TD.air_temperature.(thermo_params, cache.ᶜts)
        else
            out .= TD.air_temperature.(thermo_params, cache.ᶜts)
        end
    end,
)

###
# Potential temperature (3d)
###
add_diagnostic_variable!(
    short_name = "thetaa",
    long_name = "Air Potential Temperature",
    standard_name = "air_potential_temperature",
    units = "K",
    comments = "Potential temperature of air",
    compute! = (out, state, cache, time) -> begin
        thermo_params = CAP.thermodynamics_params(cache.params)
        if isnothing(out)
            return TD.dry_pottemp.(thermo_params, cache.ᶜts)
        else
            out .= TD.dry_pottemp.(thermo_params, cache.ᶜts)
        end
    end,
)

###
# Air pressure (3d)
###
add_diagnostic_variable!(
    short_name = "pfull",
    long_name = "Pressure at Model Full-Levels",
    units = "Pa",
    comments = "Pressure of air",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(cache.ᶜp)
        else
            out .= cache.ᶜp
        end
    end,
)

###
# Vorticity (3d)
###
add_diagnostic_variable!(
    short_name = "rv",
    long_name = "Relative Vorticity",
    standard_name = "relative_vorticity",
    units = "s^-1",
    comments = "Vertical component of relative vorticity",
    compute! = (out, state, cache, time) -> begin
        vort = @. Geometry.WVector(curlₕ(state.c.uₕ)).components.data.:1
        # We need to ensure smoothness, so we call DSS
        Spaces.weighted_dss!(vort)
        if isnothing(out)
            return copy(vort)
        else
            out .= vort
        end
    end,
)


###
# Relative humidity (3d)
###
compute_hur!(out, state, cache, time) =
    compute_hur!(out, state, cache, time, cache.atmos.moisture_model)
compute_hur!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("hur", model)

function compute_hur!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.relative_humidity.(thermo_params, cache.ᶜts)
    else
        out .= TD.relative_humidity.(thermo_params, cache.ᶜts)
    end
end

add_diagnostic_variable!(
    short_name = "hur",
    long_name = "Relative Humidity",
    standard_name = "relative_humidity",
    units = "",
    comments = "Total amount of water vapor in the air relative to the amount achievable by saturation at the current temperature",
    compute! = compute_hur!,
)

###
# Total specific humidity (3d)
###
compute_hus!(out, state, cache, time) =
    compute_hus!(out, state, cache, time, cache.atmos.moisture_model)
compute_hus!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("hus", model)

function compute_hus!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    if isnothing(out)
        return state.c.ρq_tot ./ state.c.ρ
    else
        out .= state.c.ρq_tot ./ state.c.ρ
    end
end

add_diagnostic_variable!(
    short_name = "hus",
    long_name = "Specific Humidity",
    standard_name = "specific_humidity",
    units = "kg kg^-1",
    comments = "Mass of all water phases per mass of air",
    compute! = compute_hus!,
)

###
# Surface specific humidity (2d)
###
compute_hussfc!(out, state, cache, time) =
    compute_hussfc!(out, state, cache, time, cache.atmos.moisture_model)
compute_hussfc!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("hussfc", model)

function compute_hussfc!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    thermo_params = CAP.thermodynamics_params(cache.params)
    if isnothing(out)
        return TD.total_specific_humidity.(
            thermo_params,
            cache.sfc_conditions.ts,
        )
    else
        out .=
            TD.total_specific_humidity.(thermo_params, cache.sfc_conditions.ts)
    end
end

add_diagnostic_variable!(
    short_name = "hussfc",
    long_name = "Surface Specific Humidity",
    standard_name = "specific_humidity",
    units = "kg kg^-1",
    comments = "Mass of all water phases per mass of air in the layer infinitely close to the surface",
    compute! = compute_hussfc!,
)

###
# Surface temperature (2d)
###
add_diagnostic_variable!(
    short_name = "ts",
    long_name = "Surface Temperature",
    standard_name = "surface_temperature",
    units = "K",
    comments = "Temperature of the lower boundary of the atmosphere",
    compute! = (out, state, cache, time) -> begin
        thermo_params = CAP.thermodynamics_params(cache.params)
        if isnothing(out)
            return TD.air_temperature.(thermo_params, cache.sfc_conditions.ts)
        else
            out .= TD.air_temperature.(thermo_params, cache.sfc_conditions.ts)
        end
    end,
)

###
# Eastward and northward surface drag component (2d)
###
compute_tau!(_, _, _, _, energy_form::T) where {T} =
    error_diagnostic_variable("tau", energy_form)

function compute_tau!(out, state, cache, component, energy_form::TotalEnergy)
    sfc_local_geometry =
        Fields.level(Fields.local_geometry_field(state.f), Fields.half)
    surface_ct3_unit = CT3.(unit_basis_vector_data.(CT3, sfc_local_geometry))
    (; ρ_flux_uₕ) = cache.sfc_conditions

    if isnothing(out)
        return getproperty(
            Geometry.UVVector.(
                adjoint.(ρ_flux_uₕ) .* surface_ct3_unit
            ).components.data,
            component,
        )
    else
        out .= getproperty(
            Geometry.UVVector.(
                adjoint.(ρ_flux_uₕ) .* surface_ct3_unit
            ).components.data,
            component,
        )
    end

    return
end

compute_tauu!(out, state, cache, time) =
    compute_tau!(out, state, cache, :1, cache.atmos.energy_form)
compute_tauv!(out, state, cache, time) =
    compute_tau!(out, state, cache, :2, cache.atmos.energy_form)

add_diagnostic_variable!(
    short_name = "tauu",
    long_name = "Surface Downward Eastward Wind Stress",
    standard_name = "downward_eastward_stress",
    units = "Pa",
    comments = "Eastward component of the surface drag",
    compute! = compute_tauu!,
)

add_diagnostic_variable!(
    short_name = "tauv",
    long_name = "Surface Downward Northward Wind Stress",
    standard_name = "downward_northward_stress",
    units = "Pa",
    comments = "Northward component of the surface drag",
    compute! = compute_tauv!,
)

###
# Surface energy flux (2d) - TODO: this may need to be split into sensible and latent heat fluxes
###
compute_hfes!(out, state, cache, time) =
    compute_hfes!(out, state, cache, time, cache.atmos.energy_form)
compute_hfes!(_, _, _, _, energy_form::T) where {T} =
    error_diagnostic_variable("hfes", energy_form)

function compute_hfes!(out, state, cache, time, energy_form::TotalEnergy)
    (; ρ_flux_h_tot) = cache.sfc_conditions
    sfc_local_geometry =
        Fields.level(Fields.local_geometry_field(state.f), Fields.half)
    surface_ct3_unit = CT3.(unit_basis_vector_data.(CT3, sfc_local_geometry))
    if isnothing(out)
        return dot.(ρ_flux_h_tot, surface_ct3_unit)
    else
        out .= dot.(ρ_flux_h_tot, surface_ct3_unit)
    end
end

add_diagnostic_variable!(
    short_name = "hfes",
    long_name = "Surface Upward Energy Flux",
    units = "W m^-2",
    comments = "Energy flux at the surface",
    compute! = compute_hfes!,
)

###
# Surface evaporation (2d)
###
compute_evspsbl!(out, state, cache, time) = compute_evspsbl!(
    out,
    state,
    cache,
    time,
    cache.atmos.moisture_model,
    cache.atmos.energy_form,
)
compute_evspsbl!(
    _,
    _,
    _,
    _,
    moisture_model::T1,
    energy_form::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute surface_evaporation with energy_form = TotalEnergy() and with a moist model",
)

function compute_evspsbl!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
    energy_form::TotalEnergy,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    (; ρ_flux_q_tot) = cache.sfc_conditions
    sfc_local_geometry =
        Fields.level(Fields.local_geometry_field(state.f), Fields.half)
    surface_ct3_unit = CT3.(unit_basis_vector_data.(CT3, sfc_local_geometry))

    if isnothing(out)
        return dot.(ρ_flux_q_tot, surface_ct3_unit)
    else
        out .= dot.(ρ_flux_q_tot, surface_ct3_unit)
    end
end

add_diagnostic_variable!(
    short_name = "evspsbl",
    long_name = "Evaporation Including Sublimation and Transpiration",
    units = "kg m^-2 s^-1",
    comments = "evaporation at the surface",
    compute! = compute_evspsbl!,
)

###
# Precipitation (2d) - TODO: change to kg m^-2 s^-1
###
compute_pr!(out, state, cache, time) =
    compute_pr!(out, state, cache, time, cache.atmos.precip_model)
compute_pr!(_, _, _, _, precip_model::T) where {T} =
    error_diagnostic_variable("pr", precip_model)

function compute_pr!(out, state, cache, time, precip_model::Microphysics0Moment)
    if isnothing(out)
        return cache.col_integrated_rain .+ cache.col_integrated_snow
    else
        out .= cache.col_integrated_rain .+ cache.col_integrated_snow
    end
end

add_diagnostic_variable!(
    short_name = "pr",
    long_name = "Precipitation",
    standard_name = "precipitation",
    units = "m s^-1",
    comments = "Total precipitation including rain and snow",
    compute! = compute_pr!,
)
###
# Topography
###
compute_orog!(out, state, cache, time) =
    compute_orog!(out, state, cache, time, axes(state.c).hypsography)

function compute_orog!(out, state, cache, time, hypsography::Spaces.Flat)
    # When we have a Flat topography, we just have to return a field of zeros
    if isnothing(out)
        return zeros(Spaces.horizontal_space(axes(state.c.ρ)))
    else
        # There's shouldn't be much point in this branch, but let's leave it here for
        # consistency
        out .= zeros(Spaces.horizontal_space(axes(state.c.ρ)))
    end
end

function compute_orog!(out, state, cache, time, hypsography)
    if isnothing(out)
        return hypsography.surface
    else
        out .= hypsography.surface
    end
end

add_diagnostic_variable!(
    short_name = "orog",
    long_name = "Surface Altitude",
    standard_name = "surface_altitude",
    units = "m",
    comments = "Elevation of the horizontal coordinates",
    compute! = compute_orog!,
)
