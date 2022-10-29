function moisture_model(parsed_args)
    moisture_name = parsed_args["moist"]
    @assert moisture_name in ("dry", "equil", "nonequil")
    return if moisture_name == "dry"
        DryModel()
    elseif moisture_name == "equil"
        EquilMoistModel()
    elseif moisture_name == "nonequil"
        NonEquilMoistModel()
    end
end

function model_config(parsed_args)
    config = parsed_args["config"]
    return if config == "sphere"
        SphericalModel()
    elseif config == "column"
        SingleColumnModel()
    elseif config == "box"
        BoxModel()
    end
end

function energy_form(parsed_args)
    energy_name = parsed_args["energy_name"]
    @assert energy_name in ("rhoe", "rhoe_int", "rhotheta")
    vert_diff = parsed_args["vert_diff"]
    if vert_diff
        @assert energy_name == "rhoe"
    end
    return if energy_name == "rhoe"
        TotalEnergy()
    elseif energy_name == "rhoe_int"
        InternalEnergy()
    elseif energy_name == "rhotheta"
        PotentialTemperature()
    end
end

function compressibility_model(parsed_args)
    anelastic_dycore = parsed_args["anelastic_dycore"]
    @assert anelastic_dycore in (true, false)
    return if anelastic_dycore
        AnelasticFluid()
    else
        CompressibleFluid()
    end
end

function radiation_mode(parsed_args, ::Type{FT}) where {FT}
    radiation_name = parsed_args["rad"]
    @assert radiation_name in (
        nothing,
        "clearsky",
        "gray",
        "allsky",
        "allskywithclear",
        "DYCOMS_RF01",
        "TRMM_LBA",
    )
    return if radiation_name == "clearsky"
        RRTMGPI.ClearSkyRadiation()
    elseif radiation_name == "gray"
        RRTMGPI.GrayRadiation()
    elseif radiation_name == "allsky"
        RRTMGPI.AllSkyRadiation()
    elseif radiation_name == "allskywithclear"
        RRTMGPI.AllSkyRadiationWithClearSkyDiagnostics()
    elseif radiation_name == "DYCOMS_RF01"
        RadiationDYCOMS_RF01{FT}()
    elseif radiation_name == "TRMM_LBA"
        RadiationTRMM_LBA(FT)
    else
        nothing
    end
end

function precipitation_model(parsed_args)
    precip_model = parsed_args["precip_model"]
    return if precip_model == nothing
        NoPrecipitation()
    elseif precip_model == "0M"
        Microphysics0Moment()
    elseif precip_model == "1M"
        Microphysics1Moment()
    else
        error("Invalid precip_model $(precip_model)")
    end
end

function forcing_type(parsed_args)
    forcing = parsed_args["forcing"]
    @assert forcing in (nothing, "held_suarez")
    return if forcing == nothing
        nothing
    elseif forcing == "held_suarez"
        HeldSuarezForcing()
    end
end

function subsidence_model(parsed_args, radiation_mode, FT)
    subsidence = parsed_args["subsidence"]
    subsidence == nothing && return nothing

    prof = if subsidence == "Bomex"
        APL.Bomex_subsidence(FT)
    elseif subsidence == "LifeCycleTan2018"
        APL.LifeCycleTan2018_subsidence(FT)
    elseif subsidence == "Rico"
        APL.Rico_subsidence(FT)
    elseif subsidence == "DYCOMS"
        @assert radiation_mode isa RadiationDYCOMS_RF01
        z -> -z * radiation_mode.divergence
    else
        error("Uncaught case")
    end
    return Subsidence(prof)
end

function large_scale_advection_model(parsed_args, ::Type{FT}) where {FT}
    ls_adv = parsed_args["ls_adv"]
    ls_adv == nothing && return nothing

    (prof_dTdt₀, prof_dqtdt₀) = if ls_adv == "Bomex"
        (APL.Bomex_dTdt(FT), APL.Bomex_dqtdt(FT))
    elseif ls_adv == "LifeCycleTan2018"
        (APL.LifeCycleTan2018_dTdt(FT), APL.LifeCycleTan2018_dqtdt(FT))
    elseif ls_adv == "Rico"
        (APL.Rico_dTdt(FT), APL.Rico_dqtdt(FT))
    elseif ls_adv == "ARM_SGP"
        (APL.ARM_SGP_dTdt(FT), APL.ARM_SGP_dqtdt(FT))
    elseif ls_adv == "GATE_III"
        (APL.GATE_III_dTdt(FT), APL.GATE_III_dqtdt(FT))
    else
        error("Uncaught case")
    end
    # See https://clima.github.io/AtmosphericProfilesLibrary.jl/dev/
    # for which functions accept which arguments.
    prof_dqtdt = if ls_adv in ("Bomex", "LifeCycleTan2018", "Rico", "GATE_III")
        (thermo_params, ᶜts, t, z) -> prof_dqtdt₀(z)
    elseif ls_adv == "ARM_SGP"
        (thermo_params, ᶜts, t, z) ->
            prof_dqtdt₀(TD.exner(thermo_params, ᶜts), t, z)
    end
    prof_dTdt = if ls_adv in ("Bomex", "LifeCycleTan2018", "Rico")
        (thermo_params, ᶜts, t, z) ->
            prof_dTdt₀(TD.exner(thermo_params, ᶜts), z)
    elseif ls_adv == "ARM_SGP"
        (thermo_params, ᶜts, t, z) -> prof_dTdt₀(t, z)
    elseif ls_adv == "GATE_III"
        (thermo_params, ᶜts, t, z) -> prof_dTdt₀(z)
    end

    return LargeScaleAdvection(prof_dTdt, prof_dqtdt)
end

function edmf_coriolis(parsed_args, ::Type{FT}) where {FT}
    edmf_coriolis = parsed_args["edmf_coriolis"]
    edmf_coriolis == nothing && return nothing
    (prof_u, prof_v) = if edmf_coriolis == "Bomex"
        (APL.Bomex_geostrophic_u(FT), z -> FT(0))
    elseif edmf_coriolis == "LifeCycleTan2018"
        (APL.LifeCycleTan2018_geostrophic_u(FT), z -> FT(0))
    elseif edmf_coriolis == "Rico"
        (APL.Rico_geostrophic_ug(FT), APL.Rico_geostrophic_vg(FT))
    elseif edmf_coriolis == "ARM_SGP"
        (z -> FT(10), z -> FT(0))
    elseif edmf_coriolis == "DYCOMS_RF01"
        (z -> FT(7), z -> FT(-5.5))
    elseif edmf_coriolis == "DYCOMS_RF02"
        (z -> FT(5), z -> FT(-5.5))
    elseif edmf_coriolis == "GABLS"
        (APL.GABLS_geostrophic_ug(FT), APL.GABLS_geostrophic_vg(FT))
    else
        error("Uncaught case")
    end

    coriolis_params = Dict()
    coriolis_params["Bomex"] = FT(0.376e-4)
    coriolis_params["LifeCycleTan2018"] = FT(0.376e-4)
    coriolis_params["Rico"] = FT(4.5e-5)
    coriolis_params["ARM_SGP"] = FT(8.5e-5)
    coriolis_params["DYCOMS_RF01"] = FT(0) # TODO: check this
    coriolis_params["DYCOMS_RF02"] = FT(0) # TODO: check this
    coriolis_params["GABLS"] = FT(1.39e-4)
    coriolis_param = coriolis_params[edmf_coriolis]
    return EDMFCoriolis(prof_u, prof_v, coriolis_param)
end

function turbconv_model(FT, moisture_model, precip_model, parsed_args, namelist)
    turbconv = parsed_args["turbconv"]
    @assert turbconv in (nothing, "edmf")
    return if turbconv == "edmf"
        TC.EDMFModel(FT, namelist, moisture_model, precip_model, parsed_args)
    else
        nothing
    end
end

function surface_scheme(FT, parsed_args)
    surface_scheme = parsed_args["surface_scheme"]
    @assert surface_scheme in (nothing, "bulk", "monin_obukhov")
    return if surface_scheme == "bulk"
        BulkSurfaceScheme()
    elseif surface_scheme == "monin_obukhov"
        MoninObukhovSurface()
    elseif surface_scheme == nothing
        surface_scheme
    end
end

"""
    ThermoDispatcher(model_spec)

A helper method for creating a thermodynamics dispatcher
from the model specification struct.
"""
function ThermoDispatcher(model_spec)
    (; energy_form, moisture_model, compressibility_model) = model_spec
    return ThermoDispatcher(;
        energy_form,
        moisture_model,
        compressibility_model,
    )
end
