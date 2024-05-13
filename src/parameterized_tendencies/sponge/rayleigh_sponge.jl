#####
##### Rayleigh sponge
#####

import ClimaCore.Fields as Fields

rayleigh_sponge_cache(Y, atmos::AtmosModel) =
    rayleigh_sponge_cache(Y, atmos.rayleigh_sponge)

rayleigh_sponge_cache(Y, ::Nothing) = (;)
rayleigh_sponge_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function rayleigh_sponge_cache(Y, rs::RayleighSponge)
    FT = Spaces.undertype(axes(Y.c))
    (; zd, α_uₕ, α_w) = rs
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z
    ᶜαₘ_uₕ = @. ifelse(ᶜz > zd, α_uₕ, $(FT(0)))
    ᶠαₘ_w = @. ifelse(ᶠz > zd, α_w, $(FT(0)))
    zmax = maximum(ᶠz)
    ᶜβ_rayleigh_uₕ = @. ᶜαₘ_uₕ * sin($(FT(π)) / 2 * (ᶜz - zd) / (zmax - zd))^2
    ᶠβ_rayleigh_w = @. ᶠαₘ_w * sin($(FT(π)) / 2 * (ᶠz - zd) / (zmax - zd))^2
    return (; ᶜβ_rayleigh_uₕ, ᶠβ_rayleigh_w)
end

function rayleigh_sponge_tendency!(Yₜ, Y, p, t, ::RayleighSponge)
    (; ᶜβ_rayleigh_uₕ) = p.rayleigh_sponge
    @. Yₜ.c.uₕ -= ᶜβ_rayleigh_uₕ * Y.c.uₕ
end
