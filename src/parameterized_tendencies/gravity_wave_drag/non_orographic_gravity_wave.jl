#####
##### Non-orographic gravity wave parameterization
#####

import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Operators as Operators

non_orographic_gravity_wave_cache(Y, atmos::AtmosModel) =
    non_orographic_gravity_wave_cache(
        Y,
        atmos.non_orographic_gravity_wave,
        atmos.model_config,
    )

non_orographic_gravity_wave_cache(Y, ::Nothing, ::AbstractModelConfig) = (;)

non_orographic_gravity_wave_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function non_orographic_gravity_wave_cache(
    Y,
    gw::NonOrographyGravityWave,
    ::SingleColumnModel,
)
    FT = Spaces.undertype(axes(Y.c))
    (; source_height, Bw, Bn, Bt_0, dc, cmax, c0, nk, cw, cn) = gw

    nc = Int(floor(FT(2 * cmax / dc + 1)))
    c = [FT((n - 1) * dc - cmax) for n in 1:nc]
    source_level_z = similar(Fields.level(Y.c.ρ, 1), Tuple{FT, FT})
    ᶜlevel = similar(Y.c.ρ, FT)
    for i in 1:Spaces.nlevels(axes(Y.c.ρ))
        fill!(Fields.level(ᶜlevel, i), i)
    end
    damp_level_z = similar(source_level_z)
    return (;
        gw_source_height = source_height,
        gw_source_ampl = Bt_0 .* ones(FT, axes(Fields.level(Y.c.ρ, 1))),
        gw_Bw = Bw .* ones(FT, axes(Fields.level(Y.c.ρ, 1))),
        gw_Bn = Bn .* ones(FT, axes(Fields.level(Y.c.ρ, 1))),
        gw_B0 = similar(c),
        gw_c = c,
        gw_cw = cw .* ones(FT, axes(Fields.level(Y.c.ρ, 1))),
        gw_cn = cn .* ones(FT, axes(Fields.level(Y.c.ρ, 1))),
        gw_c0 = c0,
        gw_flag = ones(FT, axes(Fields.level(Y.c.ρ, 1))),
        gw_nk = Int(nk),
        ᶜbuoyancy_frequency = similar(Y.c.ρ),
        ᶜdTdz = similar(Y.c.ρ),
        source_level_z,
        damp_level_z,
        source_level = similar(Fields.level(Y.c.ρ, 1)),
        damp_level = similar(Fields.level(Y.c.ρ, 1)),
        ᶜlevel,
        u_phy = similar(Y.c.ρ),
        v_phy = similar(Y.c.ρ),
        uforcing = similar(Y.c.ρ),
        vforcing = similar(Y.c.ρ),
    )
end

function non_orographic_gravity_wave_cache(
    Y,
    gw::NonOrographyGravityWave,
    ::SphericalModel,
)

    FT = Spaces.undertype(axes(Y.c))
    (; source_pressure, damp_pressure, Bw, Bn, Bt_0, Bt_n, Bt_s, Bt_eq) = gw
    (; ϕ0_s, ϕ0_n, dϕ_n, dϕ_s, dc, cmax, c0, nk, cw, cw_tropics, cn) = gw

    nc = Int(floor(FT(2 * cmax / dc + 1)))
    c = [FT((n - 1) * dc - cmax) for n in 1:nc]

    ᶜlocal_geometry = Fields.local_geometry_field(Fields.level(Y.c, 1))
    lat = ᶜlocal_geometry.coordinates.lat

    gw_Bn = @. ifelse(dϕ_s <= lat <= dϕ_n, FT(0), Bn)
    gw_cw = @. ifelse(dϕ_s <= lat <= dϕ_n, cw_tropics, cw)
    gw_flag = @. ifelse(dϕ_s <= lat <= dϕ_n, FT(0), FT(1))
    gw_Bw = ones(FT, axes(lat)) .* Bw
    gw_cn = ones(FT, axes(lat)) .* cn

    source_level_z = similar(Fields.level(Y.c.ρ, 1), Tuple{FT, FT})
    ᶜlevel = similar(Y.c.ρ, FT)
    for i in 1:Spaces.nlevels(axes(Y.c.ρ))
        fill!(Fields.level(ᶜlevel, i), i)
    end
    damp_level_z = similar(source_level_z)

    # This is GFDL source specs -> a smooth function
    # source_ampl = @. Bt_0 +
    #     Bt_n * FT(0.5) * (FT(1) + tanh((lat - ϕ0_n) / dϕ_n)) +
    #     Bt_s * FT(0.5) * (FT(1) + tanh((lat - ϕ0_s) / dϕ_s))

    # This latitude depend source follows MiMA specs
    source_ampl = @. ifelse(
        (lat > ϕ0_n) | (lat < ϕ0_s),
        Bt_0 +
        Bt_n * FT(0.5) * (FT(1) + tanh((lat - ϕ0_n) / dϕ_n)) +
        Bt_s * FT(0.5) * (FT(1) + tanh((lat - ϕ0_s) / dϕ_s)),
        ifelse(
            dϕ_s <= lat <= dϕ_n,
            Bt_eq,
            ifelse(
                dϕ_n <= lat <= ϕ0_n,
                Bt_0 + (Bt_eq - Bt_0) / (ϕ0_n - dϕ_n) * (ϕ0_n - lat),
                Bt_0 + (Bt_eq - Bt_0) / (ϕ0_s - dϕ_s) * (ϕ0_s - lat),
            ),
        ),
    )

    return (;
        gw_source_pressure = source_pressure,
        gw_damp_pressure = damp_pressure,
        gw_source_ampl = source_ampl,
        gw_Bw = gw_Bw,
        gw_Bn = gw_Bn,
        gw_B0 = similar(c),
        gw_c = c,
        gw_cw = gw_cw,
        gw_cn = gw_cn,
        gw_c0 = c0,
        gw_flag = gw_flag,
        gw_nk = Int(nk),
        ᶜbuoyancy_frequency = similar(Y.c.ρ),
        ᶜdTdz = similar(Y.c.ρ),
        source_level_z,
        damp_level_z,
        source_level = similar(Fields.level(Y.c.ρ, 1)),
        damp_level = similar(Fields.level(Y.c.ρ, 1)),
        ᶜlevel,
        u_phy = similar(Y.c.ρ),
        v_phy = similar(Y.c.ρ),
        uforcing = similar(Y.c.ρ),
        vforcing = similar(Y.c.ρ),
    )
end

function non_orographic_gravity_wave_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::NonOrographyGravityWave,
)
    #unpack
    (; ᶜT,) = p.core
    (; ᶜts) = p.precomputed
    (; params) = p
    (;
        ᶜdTdz,
        ᶜbuoyancy_frequency,
        source_level,
        source_level_z,
        damp_level,
        damp_level_z,
        u_phy,
        v_phy,
        uforcing,
        vforcing,
        ᶜlevel,
    ) = p.non_orographic_gravity_wave
    (; model_config) = p.atmos
    (;
        gw_source_ampl,
        gw_Bw,
        gw_Bn,
        gw_B0,
        gw_c,
        gw_cw,
        gw_cn,
        gw_flag,
        gw_c0,
        gw_nk,
    ) = p.non_orographic_gravity_wave

    if model_config isa SingleColumnModel
        (; gw_source_height) = p.non_orographic_gravity_wave
    elseif model_config isa SphericalModel
        (; gw_source_pressure, gw_damp_pressure) = p.non_orographic_gravity_wave
    end
    ᶜρ = Y.c.ρ
    ᶜz = Fields.coordinate_field(Y.c).z
    FT = Spaces.undertype(axes(Y.c))
    # parameters
    thermo_params = CAP.thermodynamics_params(params)
    grav = CAP.grav(params)

    # compute buoyancy frequency
    @. ᶜT = TD.air_temperature(thermo_params, ᶜts)

    ᶜdTdz .= Geometry.WVector.(ᶜgradᵥ.(ᶠinterp.(ᶜT))).components.data.:1

    @. ᶜbuoyancy_frequency =
        (grav / ᶜT) * (ᶜdTdz + grav / TD.cp_m(thermo_params, ᶜts))
    ᶜbuoyancy_frequency = @. ifelse(
        ᶜbuoyancy_frequency < FT(2.5e-5),
        FT(sqrt(2.5e-5)),
        sqrt(abs(ᶜbuoyancy_frequency)),
    ) # to avoid small numbers

    if model_config isa SingleColumnModel
        # source level: the index of the level that is closest to the source height

        Operators.column_mapreduce!(
            min_distance_reduce,
            source_level_z,
            ᶜz,
            ᶜlevel,
        ) do z, level
            (abs.(z .- gw_source_height), level)
        end
        source_level = source_level_z.:2

        fill!(damp_level, Spaces.nlevels(axes(ᶜz)))

    elseif model_config isa SphericalModel
        (; ᶜp) = p.precomputed
        # source level: the index of the highest level whose pressure is higher than source pressure

        Operators.column_mapreduce!(
            positive_selector_reduce,
            source_level_z,
            ᶜp,
            ᶜlevel,
        ) do p, level
            (p .- gw_source_pressure, level)
        end
        source_level = source_level_z.:2


        # damp level: the index of the lowest level whose pressure is lower than the damp pressure

        Operators.column_mapreduce!(
            negative_selector_reduce,
            damp_level_z,
            ᶜp,
            ᶜlevel,
        ) do p, level
            (p .- gw_damp_pressure, level)
        end
        damp_level = damp_level_z.:2

    end

    # prepare physical uv input variables for gravity_wave_forcing()
    u_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:1
    v_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:2

    # GW parameterization applied bycolume
    Fields.bycolumn(axes(ᶜρ)) do colidx
        parent(uforcing[colidx]) .= non_orographic_gravity_wave_forcing(
            vec(parent(u_phy[colidx])),
            vec(parent(ᶜbuoyancy_frequency[colidx])),
            vec(parent(ᶜρ[colidx])),
            vec(parent(ᶜz[colidx])),
            Int(parent(source_level[colidx])[1]),
            Int(parent(damp_level[colidx])[1]),
            parent(gw_source_ampl[colidx])[1],
            parent(gw_Bw[colidx])[1],
            parent(gw_Bn[colidx])[1],
            gw_B0,
            parent(gw_cw[colidx])[1],
            parent(gw_cn[colidx])[1],
            parent(gw_flag[colidx])[1],
            gw_c,
            gw_c0,
            gw_nk,
        )

        parent(vforcing[colidx]) .= non_orographic_gravity_wave_forcing(
            vec(parent(v_phy[colidx])),
            vec(parent(ᶜbuoyancy_frequency[colidx])),
            vec(parent(ᶜρ[colidx])),
            vec(parent(ᶜz[colidx])),
            Int(parent(source_level[colidx])[1]),
            Int(parent(damp_level[colidx])[1]),
            parent(gw_source_ampl[colidx])[1],
            parent(gw_Bw)[1],
            parent(gw_Bn)[1],
            gw_B0,
            parent(gw_cw)[1],
            parent(gw_cn)[1],
            parent(gw_flag)[1],
            gw_c,
            gw_c0,
            gw_nk,
        )

    end

    # physical uv forcing converted to Covariant12Vector and added up to uₕ tendencies
    @. Yₜ.c.uₕ +=
        Geometry.Covariant12Vector.(Geometry.UVVector.(uforcing, vforcing))
    return nothing
end

function non_orographic_gravity_wave_forcing(
    old_ᶜu,
    old_ᶜbf,
    old_ᶜρ,
    old_ᶜz,
    source_level,
    damp_level,
    source_ampl,
    Bw,
    Bn,
    B0,
    cw,
    cn,
    flag,
    c,
    c0,
    nk,
)
    FT = eltype(old_ᶜz)
    # add an extra layer above model top so that forcing between the very top
    # model layer and the upper boundary can be calculated
    ᶜu = vcat(old_ᶜu, FT(2) * old_ᶜu[end] - old_ᶜu[end - 1])
    ᶜρ = vcat(old_ᶜρ, old_ᶜρ[end] * old_ᶜρ[end] / old_ᶜρ[end - 1])
    ᶜbf = vcat(old_ᶜbf, old_ᶜbf[end])
    ᶜz = vcat(old_ᶜz, FT(2) * old_ᶜz[end] - old_ᶜz[end - 1])

    # wave spectra and the source amplitude
    nc = length(c)
    c_hat0 = c .- ᶜu[source_level] # c0mu0

    Bw_exp = @. exp(-log(2.0) * ((c * flag + c_hat0 * (1 - flag) - c0) / cw)^2)
    Bn_exp = @. exp(-log(2.0) * ((c * flag + c_hat0 * (1 - flag) - c0) / cn)^2)
    B0 = @. sign(c_hat0) * (Bw * Bw_exp + Bn * Bn_exp)

    Bsum = sum(abs.(B0))
    if (Bsum == 0.0)
        error("zero flux input at source level")
    end
    # intermittency
    eps = calc_intermitency(ᶜρ[source_level], source_ampl, nk, Bsum)

    # horizontal wave length
    kwv = [2.0 * π / ((30.0 * (10.0^n)) * 1.e3) for n in 1:nk]
    k2 = kwv .* kwv

    # forcing
    wave_forcing = zeros(length(ᶜu))
    gwf = zeros(length(ᶜu) - 1)
    for ink in 1:nk # loop over all wave lengths

        mask = ones(nc)  # mask to determine which waves propagate upward
        for k in source_level:length(ᶜu) # here ᶜu has one additional level above model top
            fac = FT(0.5) * (ᶜρ[k] / ᶜρ[source_level]) * kwv[ink] / ᶜbf[k]

            ᶜHb = -(ᶜz[k] - ᶜz[k - 1]) / log(ᶜρ[k] / ᶜρ[k - 1])  # density scale height
            alp2 = 0.25 / (ᶜHb * ᶜHb)
            ω_r = sqrt((ᶜbf[k] * ᶜbf[k] * k2[ink]) / (k2[ink] + alp2)) # omc: (critical frequency that marks total internal reflection)

            fm = FT(0)
            for n in 1:nc
                # check only those waves which are still propagating, i.e., mask = 1.0
                if (mask[n]) == 1.0
                    c_hat = c[n] - ᶜu[k] # c0mu
                    # f phase speed matches the wind speed, remove c(n) from the set of propagating waves.
                    if c_hat == 0.0
                        mask[n] = 0.0
                    else
                        # define the criterion which determines if wave is reflected at this level (test).
                        test = abs(c_hat) * kwv[ink] - ω_r
                        if test >= 0.0
                            # wave has undergone total internal reflection. remove it from the propagating set.
                            mask[n] = 0.0
                        else
                            if k == length(ᶜu)
                                # this is added in MiMA implementation:
                                # all momentum flux that escapes across the model top
                                # is deposited to the extra level being added so that
                                # momentum flux is conserved
                                mask[n] = 0.0
                                if k > source_level
                                    fm = fm + B0[n]
                                end
                            else
                                # if wave is not reflected at this level, determine if it is
                                # breaking at this level (Foc >= 0), or if wave speed relative to
                                # windspeed has changed sign from its value at the source level
                                # (c_hat0[n] * c_hat <= 0). if it is above the source level and is
                                # breaking, then add its momentum flux to the accumulated sum at
                                # this level.
                                # set mask=0.0 to remove phase speed band c[n] from the set of active
                                # waves moving upwards to the next level.
                                Foc = B0[n] / (c_hat)^3 - fac
                                if Foc >= 0.0 || (c_hat0[n] * c_hat <= 0.0)
                                    mask[n] = 0.0
                                    if k > source_level
                                        fm = fm + B0[n]
                                    end
                                end
                            end
                        end # (test >= 0.0)
                    end #(c_hat == 0.0)
                end # mask = 0

            end # nc: phase speed loop

            # compute the gravity wave momentum flux forcing
            # obtained across the entire wave spectrum at this level.
            if k > source_level
                rbh = sqrt(ᶜρ[k] * ᶜρ[k - 1])
                wave_forcing[k] =
                    (ᶜρ[source_level] / rbh) * fm * eps / (ᶜz[k] - ᶜz[k - 1])
                if k == length(ᶜu)
                    wave_forcing[k - 1] = 0.5 * wave_forcing[k - 1]
                else
                    wave_forcing[k - 1] =
                        0.5 * (wave_forcing[k - 1] + wave_forcing[k])
                end
            else
                wave_forcing[k] = 0.0
            end

        end # k

        # model top: deposit remaining momentum flux that goes across the model top
        # to the levels above the damp level
        # This is not included in Joan Alexander's original code nor the GFDL implementation;
        # but is added in MiMA based on Tiffany Shaw's paper:
        # https://journals.ametsoc.org/view/journals/clim/22/10/2009jcli2688.1.xml?tab_body=pdf
        for k in damp_level:(length(ᶜu) - 1)
            wave_forcing[k] =
                wave_forcing[k] +
                wave_forcing[end] / (length(ᶜu) + 1 - damp_level)
        end

        # forcing
        for k in source_level:(length(ᶜu) - 1)
            gwf[k] = gwf[k] + wave_forcing[k]
        end

    end # ink

    return gwf
end

# calculate the intermittency factor eps -> assuming constant Δc.

function calc_intermitency(ρ_source_level, source_ampl, nk, Bsum)
    return (source_ampl / ρ_source_level / nk) / Bsum
end

@inline min_distance_reduce(a, b) = ifelse(a[1] < b[1], a, b)
@inline positive_selector_reduce(a, b) = ifelse(b[1] <= 0, a, b)
@inline negative_selector_reduce(a, b) = ifelse(a[1] >= 0, b, a)
