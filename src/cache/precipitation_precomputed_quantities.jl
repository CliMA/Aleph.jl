#####
##### Precomputed quantities
#####
import CloudMicrophysics.Microphysics1M as CM1
import ClimaCore.Spaces as Spaces

"""
    set_precipitation_precomputed_quantities!(Y, p, t)

Updates the precipitation terminal velocity stored in `p`
for the 1-moment microphysics scheme
"""
function set_precipitation_precomputed_quantities!(Y, p, t, ::Microphysics1Moment)
    @assert (p.atmos.precip_model isa Microphysics1Moment)

    (; ᶜwᵣ, ᶜwₛ) = p.precomputed

    cmp = CAP.microphysics_params(p.params)

    # compute the precipitation terminal velocity [m/s]
    @. ᶜwᵣ = CM1.terminal_velocity(
        cmp.pr,
        cmp.tv.rain,
        Y.c.ρ,
        abs(Y.c.ρq_rai / Y.c.ρ),
    )
    @. ᶜwₛ = CM1.terminal_velocity(
        cmp.ps,
        cmp.tv.snow,
        Y.c.ρ,
        abs(Y.c.ρq_sno / Y.c.ρ),
    )
    return nothing
end

function set_precipitation_precomputed_quantities!(Y, p, t, ::MicrophysicsNMoment)
    @assert (p.atmos.precip_model isa MicrophysicsNMoment) # TODO Nmoment
    (; ᶜwᵣ) = p.precomputed
    FT = Spaces.undertype(axes(Y.c))
    # compute the precipitation terminal velocity [m/s]
    @. ᶜwᵣ = FT(0) # TODO Nmoment
    return nothing
end
