#####
##### Radiation
#####

import ClimaComms
import ClimaCore: Device, DataLayouts, Geometry, Spaces, Fields, Operators
import Insolation
import Thermodynamics as TD
import .Parameters as CAP
import RRTMGP
import .RRTMGPInterface as RRTMGPI

using Dierckx: Spline1D
using StatsBase: mean


radiation_model_cache(Y, atmos::AtmosModel, args...) =
    radiation_model_cache(Y, atmos.radiation_mode, args...)

#####
##### No Radiation
#####

radiation_model_cache(Y, radiation_mode::Nothing; args...) = (;)
radiation_tendency!(Yₜ, Y, p, t, colidx, ::Nothing) = nothing

#####
##### RRTMGP Radiation
#####

function radiation_model_cache(
    Y,
    radiation_mode::RRTMGPI.AbstractRRTMGPMode,
    params,
    ᶜp; # Used for ozone
    interpolation = RRTMGPI.BestFit(),
    bottom_extrapolation = RRTMGPI.SameAsInterpolation(),
    data_loader = rrtmgp_data_loader,
)
    context = ClimaComms.context(axes(Y.c))
    device = context.device
    (; idealized_h2o, idealized_insolation, idealized_clouds) = radiation_mode
    FT = Spaces.undertype(axes(Y.c))
    DA = ClimaComms.array_type(device){FT}
    rrtmgp_params = CAP.rrtmgp_params(params)
    if idealized_h2o && radiation_mode isa RRTMGPI.GrayRadiation
        error("idealized_h2o can't be used with $radiation_mode")
    end
    if idealized_clouds && (
        radiation_mode isa RRTMGPI.GrayRadiation ||
        radiation_mode isa RRTMGPI.ClearSkyRadiation
    )
        error("idealized_clouds can't be used with $radiation_mode")
    end

    bottom_coords = Fields.coordinate_field(Spaces.level(Y.c, 1))
    if eltype(bottom_coords) <: Geometry.LatLongZPoint
        latitude = RRTMGPI.field2array(bottom_coords.lat)
    else
        latitude = RRTMGPI.field2array(zero(bottom_coords.z)) # flat space is on Equator
    end
    local radiation_model
    orbital_data = Insolation.OrbitalData()
    data_loader(joinpath("atmos_state", "clearsky_as.nc")) do input_data
        if radiation_mode isa RRTMGPI.GrayRadiation
            kwargs = (;
                lapse_rate = 3.5,
                optical_thickness_parameter = (@. 7.2 +
                                                  (1.8 - 7.2) *
                                                  sind(latitude)^2),
                latitude,
            )
        else
            # the pressure and ozone concentrations are provided for each of 100
            # sites, which we average across
            n = input_data.dim["layer"]
            input_center_pressure = vec(
                mean(reshape(input_data["pres_layer"][:, :], n, :); dims = 2),
            )
            # the first values along the third dimension of the ozone concentration
            # data are the present-day values
            input_center_volume_mixing_ratio_o3 =
                vec(mean(reshape(input_data["ozone"][:, :, 1], n, :); dims = 2))

            # interpolate the ozone concentrations to our initial pressures
            pressure2ozone = Spline1D(
                input_center_pressure,
                input_center_volume_mixing_ratio_o3,
            )
            if device isa ClimaComms.CUDADevice
                fv = Fields.field_values(ᶜp)
                fld_array = DA(pressure2ozone.(Array(parent(fv))))
                data = DataLayouts.rebuild(fv, fld_array)
                ᶜvolume_mixing_ratio_o3_field = Fields.Field(data, axes(ᶜp))
            else
                ᶜvolume_mixing_ratio_o3_field = @. FT(pressure2ozone(ᶜp))
            end
            center_volume_mixing_ratio_o3 =
                RRTMGPI.field2array(ᶜvolume_mixing_ratio_o3_field)

            # the first value for each global mean volume mixing ratio is the
            # present-day value
            input_vmr(name) =
                input_data[name][1] *
                parse(FT, input_data[name].attrib["units"])
            kwargs = (;
                use_global_means_for_well_mixed_gases = true,
                center_volume_mixing_ratio_h2o = NaN, # initialize in tendency
                center_volume_mixing_ratio_o3,
                volume_mixing_ratio_co2 = input_vmr("carbon_dioxide_GM"),
                volume_mixing_ratio_n2o = input_vmr("nitrous_oxide_GM"),
                volume_mixing_ratio_co = input_vmr("carbon_monoxide_GM"),
                volume_mixing_ratio_ch4 = input_vmr("methane_GM"),
                volume_mixing_ratio_o2 = input_vmr("oxygen_GM"),
                volume_mixing_ratio_n2 = input_vmr("nitrogen_GM"),
                volume_mixing_ratio_ccl4 = input_vmr("carbon_tetrachloride_GM"),
                volume_mixing_ratio_cfc11 = input_vmr("cfc11_GM"),
                volume_mixing_ratio_cfc12 = input_vmr("cfc12_GM"),
                volume_mixing_ratio_cfc22 = input_vmr("hcfc22_GM"),
                volume_mixing_ratio_hfc143a = input_vmr("hfc143a_GM"),
                volume_mixing_ratio_hfc125 = input_vmr("hfc125_GM"),
                volume_mixing_ratio_hfc23 = input_vmr("hfc23_GM"),
                volume_mixing_ratio_hfc32 = input_vmr("hfc32_GM"),
                volume_mixing_ratio_hfc134a = input_vmr("hfc134a_GM"),
                volume_mixing_ratio_cf4 = input_vmr("cf4_GM"),
                volume_mixing_ratio_no2 = 1e-8, # not available in input_data
                latitude,
            )
            if !(radiation_mode isa RRTMGPI.ClearSkyRadiation)
                kwargs = (;
                    kwargs...,
                    center_cloud_liquid_effective_radius = 12,
                    center_cloud_ice_effective_radius = 95,
                    ice_roughness = 2,
                )
                ᶜz = Fields.coordinate_field(Y.c).z
                ᶜΔz = Fields.local_geometry_field(Y.c).∂x∂ξ.components.data.:9
                if idealized_clouds # icy cloud on top and wet cloud on bottom
                    # TODO: can we avoid using DataLayouts with this?
                    #     `ᶜis_bottom_cloud = similar(ᶜz, Bool)`
                    ᶜis_bottom_cloud = Fields.Field(
                        DataLayouts.replace_basetype(
                            Fields.field_values(ᶜz),
                            Bool,
                        ),
                        axes(Y.c),
                    ) # need to fix several ClimaCore bugs in order to simplify this
                    ᶜis_top_cloud = similar(ᶜis_bottom_cloud)
                    @. ᶜis_bottom_cloud = ᶜz > 1e3 && ᶜz < 1.5e3
                    @. ᶜis_top_cloud = ᶜz > 4e3 && ᶜz < 5e3
                    kwargs = (;
                        kwargs...,
                        center_cloud_liquid_water_path = RRTMGPI.field2array(
                            @. ifelse(ᶜis_bottom_cloud, FT(0.002) * ᶜΔz, FT(0))
                        ),
                        center_cloud_ice_water_path = RRTMGPI.field2array(
                            @. ifelse(ᶜis_top_cloud, FT(0.001) * ᶜΔz, FT(0))
                        ),
                        center_cloud_fraction = RRTMGPI.field2array(
                            @. ifelse(
                                ᶜis_bottom_cloud | ᶜis_top_cloud,
                                FT(1),
                                0 * ᶜΔz,
                            )
                        ),
                    )
                else
                    kwargs = (;
                        kwargs...,
                        center_cloud_liquid_water_path = NaN, # initialized in callback
                        center_cloud_ice_water_path = NaN, # initialized in callback
                        center_cloud_fraction = NaN, # initialized in callback
                    )
                end
            end
        end

        if RRTMGPI.requires_z(interpolation) ||
           RRTMGPI.requires_z(bottom_extrapolation)
            kwargs = (;
                kwargs...,
                center_z = RRTMGPI.field2array(Fields.coordinate_field(Y.c).z),
                face_z = RRTMGPI.field2array(Fields.coordinate_field(Y.f).z),
            )
        end

        if idealized_insolation # perpetual equinox with no diurnal cycle
            solar_zenith_angle = FT(π) / 3
            weighted_irradiance =
                @. 1360 * (1 + FT(1.2) / 4 * (1 - 3 * sind(latitude)^2)) /
                   (4 * cos(solar_zenith_angle))
        else
            solar_zenith_angle = weighted_irradiance = NaN # initialized in callback
        end

        radiation_model = RRTMGPI.RRTMGPModel(
            rrtmgp_params,
            data_loader,
            # TODO: pass FT through so that it can be controlled at the top-level
            Float64,
            context;
            ncol = length(Spaces.all_nodes(axes(Spaces.level(Y.c, 1)))),
            domain_nlay = Spaces.nlevels(axes(Y.c)),
            radiation_mode,
            interpolation,
            bottom_extrapolation,
            add_isothermal_boundary_layer = true,
            center_pressure = NaN, # initialized in callback
            center_temperature = NaN, # initialized in callback
            surface_temperature = NaN, # initialized in callback
            surface_emissivity = 1,
            direct_sw_surface_albedo = 0.38,
            diffuse_sw_surface_albedo = 0.38,
            solar_zenith_angle,
            weighted_irradiance,
            kwargs...,
        )
    end
    return (;
        idealized_insolation,
        orbital_data,
        idealized_h2o,
        idealized_clouds,
        radiation_model,
        insolation_tuple = similar(Spaces.level(Y.c, 1), Tuple{FT, FT, FT}),
        ᶠradiation_flux = similar(Y.f, Geometry.WVector{FT}),
    )
end

function radiation_tendency!(Yₜ, Y, p, t, colidx, ::RRTMGPI.AbstractRRTMGPMode)
    (; ᶠradiation_flux) = p.radiation
    if :ρθ in propertynames(Y.c)
        error("radiation_tendency! not implemented for ρθ")
    elseif :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot[colidx] -= ᶜdivᵥ(ᶠradiation_flux[colidx])
    end
    return nothing
end

#####
##### DYCOMS_RF01 radiation
#####

function radiation_model_cache(Y, radiation_mode::RadiationDYCOMS_RF01)
    FT = Spaces.undertype(axes(Y.c))
    NT = NamedTuple{(:z, :ρ, :q_tot), NTuple{3, FT}}
    return (;
        ᶜκρq = similar(Y.c, FT),
        ∫_0_∞_κρq = similar(Spaces.level(Y.c, 1), FT),
        ᶠ∫_0_z_κρq = similar(Y.f, FT),
        isoline_z_ρ_q = similar(Spaces.level(Y.c, 1), NT),
        ᶠradiation_flux = similar(Y.f, Geometry.WVector{FT}),
        net_energy_flux_toa = [Geometry.WVector(FT(0))],
        net_energy_flux_sfc = [Geometry.WVector(FT(0))],
    )
end
function radiation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    radiation_mode::RadiationDYCOMS_RF01,
)
    @assert !(p.atmos.moisture_model isa DryModel)
    @assert p.atmos.energy_form isa TotalEnergy

    (; params) = p
    (; ᶜspecific, ᶜts) = p.precomputed
    (; ᶜκρq, ∫_0_∞_κρq, ᶠ∫_0_z_κρq, isoline_z_ρ_q, ᶠradiation_flux) =
        p.radiation
    thermo_params = CAP.thermodynamics_params(params)
    cp_d = CAP.cp_d(params)
    FT = Spaces.undertype(axes(Y.c))
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z

    # TODO: According to the paper, we should replace liquid_specific_humidity
    # with TD.mixing_ratios(thermo_params, ᶜts[colidx]).liq, but this wouldn't
    # match the original code from TurbulenceConvection.
    @. ᶜκρq[colidx] =
        radiation_mode.kappa *
        Y.c.ρ[colidx] *
        TD.liquid_specific_humidity(thermo_params, ᶜts[colidx])

    Operators.column_integral_definite!(∫_0_∞_κρq[colidx], ᶜκρq[colidx])

    Operators.column_integral_indefinite!(ᶠ∫_0_z_κρq[colidx], ᶜκρq[colidx])

    # Find the values of (z, ρ, q_tot) at the q_tot = 0.008 isoline, i.e., at
    # the level whose value of q_tot is closest to 0.008.
    Operators.column_mapreduce!(
        (z, ρ, q_tot) -> (; z, ρ, q_tot),
        (nt1, nt2) ->
            abs(nt1.q_tot - FT(0.008)) < abs(nt2.q_tot - FT(0.008)) ? nt1 : nt2,
        isoline_z_ρ_q[colidx],
        ᶜz[colidx],
        Y.c.ρ[colidx],
        ᶜspecific.q_tot[colidx],
    )

    zi = isoline_z_ρ_q.z
    ρi = isoline_z_ρ_q.ρ

    # TODO: According to the paper, we should remove the ifelse condition that
    # clips the third term to 0 below zi, and we should also replace cp_d with
    # cp_m, but this wouldn't match the original code from TurbulenceConvection.
    # Note: ∫_0_z_κρq - ∫_0_∞_κρq = -∫_z_∞_κρq
    @. ᶠradiation_flux[colidx] = Geometry.WVector(
        radiation_mode.F0 * exp(ᶠ∫_0_z_κρq[colidx] - ∫_0_∞_κρq[colidx]) +
        radiation_mode.F1 * exp(-(ᶠ∫_0_z_κρq[colidx])) +
        ifelse(
            ᶠz[colidx] > zi[colidx],
            ρi[colidx] *
            cp_d *
            radiation_mode.divergence *
            radiation_mode.alpha_z *
            (
                cbrt(ᶠz[colidx] - zi[colidx])^4 / 4 +
                zi[colidx] * cbrt(ᶠz[colidx] - zi[colidx])
            ),
            FT(0),
        ),
    )

    @. Yₜ.c.ρe_tot[colidx] -= ᶜdivᵥ(ᶠradiation_flux[colidx])

    return nothing
end

#####
##### TRMM_LBA radiation
#####

function radiation_model_cache(Y, radiation_mode::RadiationTRMM_LBA)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        ᶜdTdt_rad = similar(Y.c, FT),
        net_energy_flux_toa = [Geometry.WVector(FT(0))],
        net_energy_flux_sfc = [Geometry.WVector(FT(0))],
    )
end

function radiation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    radiation_mode::RadiationTRMM_LBA,
)
    FT = Spaces.undertype(axes(Y.c))
    (; params) = p
    # TODO: get working (need to add cache / function)
    rad = radiation_mode.rad_profile
    thermo_params = CAP.thermodynamics_params(params)
    ᶜdTdt_rad = p.radiation.ᶜdTdt_rad[colidx]
    ᶜρ = Y.c.ρ[colidx]
    ᶜts_gm = p.precomputed.ᶜts[colidx]
    zc = Fields.coordinate_field(axes(ᶜρ)).z
    @. ᶜdTdt_rad = rad(FT(t), zc)
    @. Yₜ.c.ρe_tot[colidx] += ᶜρ * TD.cv_m(thermo_params, ᶜts_gm) * ᶜdTdt_rad
    return nothing
end
