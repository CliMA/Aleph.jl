#####
##### Hyperdiffusion
#####

import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces

hyperdiffusion_cache(hyperdiff::Nothing, Y) = NamedTuple()

function hyperdiffusion_cache(hyperdiff::AbstractHyperdiffusion, Y)
    FT = Spaces.undertype(axes(Y.c))
    moist_kwargs = if (:ρq_tot in propertynames(Y.c))
        (; ᶜχρq_tot = similar(Y.c, FT))
    else
        NamedTuple()
    end
    tempest_kwargs = if hyperdiff isa TempestHyperdiffusion
        (; ᶠχw_data = similar(Y.f, FT))
    else
        NamedTuple()
    end

    return (;
        ᶜχ = similar(Y.c, FT),
        moist_kwargs...,
        ᶜχuₕ = similar(Y.c, Geometry.Covariant12Vector{FT}),
        tempest_kwargs...,
    )
end

function hyperdiffusion_tendency!(Yₜ, Y, p, t)
    p.atmos.hyperdiff isa Nothing && return nothing

    NVTX.@range "hyperdiffusion tendency" color = colorant"yellow" begin
        hyperdiffusion_tendency!(Yₜ, Y, p, t, p.atmos.hyperdiff)
    end
end

function hyperdiffusion_tendency!(Yₜ, Y, p, t, ::ClimaHyperdiffusion)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    divₕ = Operators.Divergence()
    wdivₕ = Operators.WeakDivergence()
    gradₕ = Operators.Gradient()
    wgradₕ = Operators.WeakGradient()
    curlₕ = Operators.Curl()
    wcurlₕ = Operators.WeakCurl()

    (; ᶜp, ᶜχ, ᶜχuₕ) = p # assume ᶜp has been updated
    (; ghost_buffer) = p
    (; κ₄, divergence_damping_factor) = p.atmos.hyperdiff
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)
    is_ρq_tot =
        :ρq_tot in propertynames(Y.c) &&
        q_tot_hyperdiffusion_enabled(p.atmos.hyperdiff)
    is_2d_pt = point_type <: Geometry.Abstract2DPoint
    is_3d_pt = !is_2d_pt

    ᶜρs =
        :ρe_tot in propertynames(Y.c) ? Y.c.ρe_tot :
        (:ρe_int in propertynames(Y.c) ? Y.c.ρe_int : Y.c.ρθ)
    ᵗρs =
        :ρe_tot in propertynames(Y.c) ? Yₜ.c.ρe_tot :
        (:ρe_int in propertynames(Y.c) ? Yₜ.c.ρe_int : Yₜ.c.ρθ)

    (:ρθ in propertynames(Y.c)) && (@. ᶜχ = wdivₕ(gradₕ(ᶜρs / ᶜρ)))
    !(:ρθ in propertynames(Y.c)) && (@. ᶜχ = wdivₕ(gradₕ((ᶜρs + ᶜp) / ᶜρ)))

    if is_ρq_tot
        (; ᶜχρq_tot) = p
        @. ᶜχρq_tot = wdivₕ(gradₕ(Y.c.ρq_tot / ᶜρ))
    end

    is_3d_pt && (@. ᶜχuₕ =
        wgradₕ(divₕ(ᶜuₕ)) - Geometry.project(
            Geometry.Covariant12Axis(),
            wcurlₕ(Geometry.project(Geometry.Covariant3Axis(), curlₕ(ᶜuₕ))),
        ))
    is_2d_pt && (@. ᶜχuₕ =
        Geometry.project(Geometry.Covariant12Axis(), wgradₕ(divₕ(ᶜuₕ))))

    NVTX.@range "dss_hyperdiffusion_tendency" color = colorant"green" begin
        Spaces.weighted_dss_start2!(ᶜχ, ghost_buffer.χ)
        is_ρq_tot &&
            (Spaces.weighted_dss_start2!(ᶜχρq_tot, ghost_buffer.ᶜχρq_tot))
        Spaces.weighted_dss_start2!(ᶜχuₕ, ghost_buffer.χuₕ)

        Spaces.weighted_dss_internal2!(ᶜχ, ghost_buffer.χ)
        is_ρq_tot &&
            (Spaces.weighted_dss_internal2!(ᶜχρq_tot, ghost_buffer.ᶜχρq_tot))
        Spaces.weighted_dss_internal2!(ᶜχuₕ, ghost_buffer.χuₕ)

        Spaces.weighted_dss_ghost2!(ᶜχ, ghost_buffer.χ)
        is_ρq_tot &&
            (Spaces.weighted_dss_ghost2!(ᶜχρq_tot, ghost_buffer.ᶜχρq_tot))
        Spaces.weighted_dss_ghost2!(ᶜχuₕ, ghost_buffer.χuₕ)
    end

    @. ᵗρs -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
    if is_ρq_tot
        @. Yₜ.c.ρq_tot -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχρq_tot))
        @. Yₜ.c.ρ -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχρq_tot))
    end

    @info "Debug: hyperdiffusion_tendency Yₜ at time $t"
    @show @. κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχρq_tot))
    @show Yₜ.c.ρq_tot
    @show @. Yₜ.c.ρ

    if is_3d_pt
        @. Yₜ.c.uₕ -=
            κ₄ * (
                divergence_damping_factor * wgradₕ(divₕ(ᶜχuₕ)) -
                Geometry.project(
                    Geometry.Covariant12Axis(),
                    wcurlₕ(
                        Geometry.project(
                            Geometry.Covariant3Axis(),
                            curlₕ(ᶜχuₕ),
                        ),
                    ),
                )
            )
    elseif is_2d_pt
        @. Yₜ.c.uₕ -=
            κ₄ *
            divergence_damping_factor *
            Geometry.Covariant12Vector(wgradₕ(divₕ(ᶜχuₕ)))
    end
    return nothing
end

function hyperdiffusion_tendency!(Yₜ, Y, p, t, ::TempestHyperdiffusion)

    divₕ = Operators.Divergence()
    wdivₕ = Operators.WeakDivergence()
    gradₕ = Operators.Gradient()
    wgradₕ = Operators.WeakGradient()
    curlₕ = Operators.Curl()
    wcurlₕ = Operators.WeakCurl()

    !(:ρθ in propertynames(Y.c)) &&
        (error("TempestHyperdiffusion is only compatible with ρθ"))
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    (; ᶜp, ᶜχ, ᶜχuₕ) = p # assume ᶜp has been updated
    (; ghost_buffer) = p
    (; κ₄, divergence_damping_factor) = p.atmos.hyperdiff
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)
    is_ρq_tot = :ρq_tot in propertynames(Y.c)
    is_2d_pt = point_type <: Geometry.Abstract2DPoint
    is_3d_pt = !is_2d_pt

    @. ᶜχ = wdivₕ(gradₕ(ᶜρ)) # ᶜχρ
    Spaces.weighted_dss2!(ᶜχ, ghost_buffer.χ)
    @. Yₜ.c.ρ -= κ₄ * wdivₕ(gradₕ(ᶜχ))

    @. ᶜχ = wdivₕ(gradₕ(Y.c.ρθ)) # ᶜχρθ
    Spaces.weighted_dss2!(ᶜχ, ghost_buffer.χ)
    @. Yₜ.c.ρθ -= κ₄ * wdivₕ(gradₕ(ᶜχ))

    (; ᶠχw_data) = p
    @. ᶠχw_data = wdivₕ(gradₕ(Y.f.w.components.data.:1))
    Spaces.weighted_dss2!(ᶠχw_data, ghost_buffer.χ)
    @. Yₜ.f.w.components.data.:1 -= κ₄ * wdivₕ(gradₕ(ᶠχw_data))

    if is_ρq_tot
        (; ᶜχρq_tot) = p
        @. ᶜχρq_tot = wdivₕ(gradₕ(Y.c.ρq_tot / ᶜρ))
        Spaces.weighted_dss2!(ᶜχρq_tot, ghost_buffer.χ)
        @. Yₜ.c.ρq_tot -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχρq_tot))
        @. Yₜ.c.ρ -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχρq_tot))
    end

    if is_3d_pt
        @. ᶜχuₕ =
            wgradₕ(divₕ(ᶜuₕ)) - Geometry.Covariant12Vector(
                wcurlₕ(Geometry.Covariant3Vector(curlₕ(ᶜuₕ))),
            )
        Spaces.weighted_dss2!(ᶜχuₕ, ghost_buffer.χuₕ)
        @. Yₜ.c.uₕ -=
            κ₄ * (
                divergence_damping_factor * wgradₕ(divₕ(ᶜχuₕ)) -
                Geometry.Covariant12Vector(
                    wcurlₕ(Geometry.Covariant3Vector(curlₕ(ᶜχuₕ))),
                )
            )
    elseif is_2d_pt
        @. ᶜχuₕ = Geometry.Covariant12Vector(wgradₕ(divₕ(ᶜuₕ)))
        Spaces.weighted_dss2!(ᶜχuₕ, ghost_buffer.χuₕ)
        @. Yₜ.c.uₕ -=
            κ₄ *
            divergence_damping_factor *
            Geometry.Covariant12Vector(wgradₕ(divₕ(ᶜχuₕ)))
    end
    return nothing
end
