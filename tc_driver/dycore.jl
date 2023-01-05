import LinearAlgebra as LA
import LinearAlgebra: ×

import ClimaAtmos as CA
import ClimaAtmos.TurbulenceConvection as TC
import ClimaAtmos.TurbulenceConvection.Parameters as TCP
const APS = TCP.AbstractTurbulenceConvectionParameters
import Thermodynamics as TD
import ClimaCore as CC
import ClimaCore.Geometry as CCG
import OrdinaryDiffEq as ODE

import CLIMAParameters as CP

include(joinpath(@__DIR__, "dycore_variables.jl"))

#####
##### Methods
#####

####
#### Reference state
####

function set_thermo_state_pθq!(Y, p, colidx)
    (; edmf_cache, params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; moisture_model) = edmf_cache.edmf
    ᶜts_gm = p.ᶜts[colidx]
    ᶜρ = Y.c.ρ[colidx]
    ᶜp = p.ᶜp[colidx]
    θ_liq_ice = edmf_cache.aux.cent.θ_liq_ice[colidx]

    if moisture_model isa CA.DryModel
        @. ᶜts_gm = TD.PhaseDry_pθ(thermo_params, ᶜp, θ_liq_ice)
    elseif moisture_model isa CA.EquilMoistModel
        ρq_tot = Y.c.ρq_tot[colidx]
        @. ᶜts_gm = TD.PhaseEquil_pθq(thermo_params, ᶜp, θ_liq_ice, ρq_tot / ᶜρ)
    else
        error("TODO: add non-equilibrium moisture model support")
    end
    nothing
end

function set_grid_mean_from_thermo_state!(thermo_params, state, grid)
    Ic = CCO.InterpolateF2C()
    If = CCO.InterpolateC2F(bottom = CCO.Extrapolate(), top = CCO.Extrapolate())
    ts_gm = TC.center_aux_grid_mean_ts(state)
    prog_gm = TC.center_prog_grid_mean(state)
    prog_gm_f = TC.face_prog_grid_mean(state)
    aux_gm = TC.center_aux_grid_mean(state)
    aux_gm_f = TC.face_aux_grid_mean(state)
    prog_gm_uₕ = TC.grid_mean_uₕ(state)

    @. prog_gm.ρ = TD.air_density(thermo_params, ts_gm)
    ρ_c = prog_gm.ρ
    ρ_f = aux_gm_f.ρ

    @. prog_gm.ρe_tot =
        ρ_c * TD.total_energy(
            thermo_params,
            ts_gm,
            LinearAlgebra.norm_sqr(prog_gm_uₕ) +
            Ic(LinearAlgebra.norm_sqr(prog_gm_f.w)) / 2,
            TC.geopotential(thermo_params, grid.zc.z),
        )

    @. prog_gm.ρq_tot = ρ_c * aux_gm.q_tot
    @. ρ_f = If(ρ_c)

    return nothing
end

function assign_thermo_aux!(state, grid, moisture_model, thermo_params)
    If = CCO.InterpolateC2F(bottom = CCO.Extrapolate(), top = CCO.Extrapolate())
    aux_gm = TC.center_aux_grid_mean(state)
    aux_gm_f = TC.face_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    ᶜts = TC.center_aux_grid_mean_ts(state)
    p_c = TC.center_aux_grid_mean_p(state)
    ρ_c = prog_gm.ρ
    ρ_f = aux_gm_f.ρ
    @. ρ_f = If(ρ_c)

    @. aux_gm.q_tot = prog_gm.ρq_tot / ρ_c
    @. aux_gm.q_liq = TD.liquid_specific_humidity(thermo_params, ᶜts)
    @. aux_gm.q_ice = TD.ice_specific_humidity(thermo_params, ᶜts)
    @. aux_gm.T = TD.air_temperature(thermo_params, ᶜts)
    @. aux_gm.RH = TD.relative_humidity(thermo_params, ᶜts)
    @. aux_gm.θ_liq_ice = TD.liquid_ice_pottemp(thermo_params, ᶜts)
    @. aux_gm.h_tot =
        TD.total_specific_enthalpy(thermo_params, ᶜts, prog_gm.ρe_tot / ρ_c)
    @. p_c = TD.air_pressure(thermo_params, ᶜts)
    @. aux_gm.θ_virt = TD.virtual_pottemp(thermo_params, ᶜts)
    return
end

function compute_implicit_gm_tendencies!(
    edmf::TC.EDMFModel,
    grid::TC.Grid,
    state::TC.State,
    surf,
    param_set::APS,
)
    tendencies_gm = TC.center_tendencies_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    aux_gm_f = TC.face_aux_grid_mean(state)
    ρ_c = prog_gm.ρ
    tendencies_gm_uₕ = TC.tendencies_grid_mean_uₕ(state)

    TC.compute_sgs_flux!(edmf, grid, state, surf, param_set)

    ∇sgs = CCO.DivergenceF2C()
    @. tendencies_gm.ρe_tot += -∇sgs(aux_gm_f.sgs_flux_h_tot)
    @. tendencies_gm.ρq_tot += -∇sgs(aux_gm_f.sgs_flux_q_tot)
    @. tendencies_gm_uₕ += -∇sgs(aux_gm_f.sgs_flux_uₕ) / ρ_c

    return nothing
end

function compute_explicit_gm_tendencies!(
    edmf::TC.EDMFModel,
    grid::TC.Grid,
    state::TC.State,
    surf,
    param_set::APS,
)
    tendencies_gm = TC.center_tendencies_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    aux_en = TC.center_aux_environment(state)
    aux_bulk = TC.center_aux_bulk(state)
    ρ_c = prog_gm.ρ
    aux_tc = TC.center_aux_turbconv(state)

    # Apply precipitation tendencies
    @. tendencies_gm.ρq_tot += ρ_c * aux_tc.qt_tendency_precip_sinks
    @. tendencies_gm.ρe_tot += ρ_c * aux_tc.e_tot_tendency_precip_sinks

    return nothing
end
