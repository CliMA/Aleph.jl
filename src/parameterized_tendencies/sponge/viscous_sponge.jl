#####
##### Viscous sponge
#####

import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Spaces as Spaces

viscous_sponge_cache(Y, atmos::AtmosModel) =
    viscous_sponge_cache(Y, atmos.viscous_sponge)

viscous_sponge_cache(Y, ::Nothing) = (;)
viscous_sponge_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function viscous_sponge_cache(Y, viscous_sponge::ViscousSponge)
    (; κ₂, zd) = viscous_sponge
    FT = Spaces.undertype(axes(Y.c))
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z
    ᶜαₘ = @. ifelse(ᶜz > zd, κ₂, FT(0))
    ᶠαₘ = @. ifelse(ᶠz > zd, κ₂, FT(0))
    zmax = maximum(ᶠz)
    ᶜβ_viscous = @. ᶜαₘ * sin(FT(π) / 2 * (ᶜz - zd) / (zmax - zd))^2
    ᶠβ_viscous = @. ᶠαₘ * sin(FT(π) / 2 * (ᶠz - zd) / (zmax - zd))^2
    return (; ᶜβ_viscous, ᶠβ_viscous)
end

function viscous_sponge_tendency!(Yₜ, Y, p, t, ::ViscousSponge)
    (; ᶜβ_viscous, ᶠβ_viscous) = p.viscous_sponge
    (; ᶜh_tot, ᶜspecific) = p.precomputed
    ᶜuₕ = Y.c.uₕ
    @. Yₜ.c.uₕ +=
        ᶜβ_viscous * (
            wgradₕ(divₕ(ᶜuₕ)) - Geometry.project(
                Geometry.Covariant12Axis(),
                wcurlₕ(Geometry.project(Geometry.Covariant3Axis(), curlₕ(ᶜuₕ))),
            )
        )
    @. Yₜ.f.u₃.components.data.:1 +=
        ᶠβ_viscous * wdivₕ(gradₕ(Y.f.u₃.components.data.:1))

    @. Yₜ.c.ρe_tot += ᶜβ_viscous * wdivₕ(Y.c.ρ * gradₕ(ᶜh_tot))
    for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue
        @. ᶜρχₜ += ᶜβ_viscous * wdivₕ(Y.c.ρ * gradₕ(ᶜχ))
        @. Yₜ.c.ρ += ᶜβ_viscous * wdivₕ(Y.c.ρ * gradₕ(ᶜχ))
    end
end
