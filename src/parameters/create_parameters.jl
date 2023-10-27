import CLIMAParameters as CP
import RRTMGP.Parameters as RP
import SurfaceFluxes as SF
import SurfaceFluxes.UniversalFunctions as UF
import Insolation.Parameters as IP
import Thermodynamics as TD
import CloudMicrophysics as CM

function create_parameter_set(config::AtmosConfig)
    # Helper function that creates a parameter struct. If a struct has nested
    # parameter structs, they must be passed to subparam_structs as a NamedTuple.
    function create_parameter_struct(param_struct; subparam_structs = (;))
        aliases = string.(fieldnames(param_struct))
        aliases = setdiff(aliases, string.(propertynames(subparam_structs)))
        pairs = CP.get_parameter_values!(toml_dict, aliases)
        # Workaround for setting τ_precip = dt
        if parsed_args["override_τ_precip"] &&
           param_struct == CM.Parameters.CloudMicrophysicsParameters
            pairs = (;
                pairs...,
                τ_precip = FT(CA.time_to_seconds(parsed_args["dt"])),
            )
        end
        return param_struct{FT, typeof.(values(subparam_structs))...}(;
            pairs...,
            subparam_structs...,
        )
    end

    (; toml_dict, parsed_args) = config
    FT = CP.float_type(toml_dict)

    thermo_params =
        create_parameter_struct(TD.Parameters.ThermodynamicsParameters)
    rrtmgp_params = create_parameter_struct(RP.RRTMGPParameters)
    insolation_params = create_parameter_struct(IP.InsolationParameters)
    microphys_params = create_parameter_struct(
        CM.Parameters.CloudMicrophysicsParameters;
        subparam_structs = (; thermo_params),
    )

    aliases = [
        "Pr_0_Businger",
        "a_m_Businger",
        "a_h_Businger",
        "ζ_a_Businger",
        "γ_Businger",
    ]
    pairs = CP.get_parameter_values!(toml_dict, aliases, "UniversalFunctions")
    pairs = (; pairs...) # convert to NamedTuple
    pairs = (;
        Pr_0 = pairs.Pr_0_Businger,
        a_m = pairs.a_m_Businger,
        a_h = pairs.a_h_Businger,
        ζ_a = pairs.ζ_a_Businger,
        γ = pairs.γ_Businger,
    )
    ufp = UF.BusingerParams{FT}(; pairs...)

    surf_flux_params = create_parameter_struct(
        SF.Parameters.SurfaceFluxesParameters;
        subparam_structs = (; ufp, thermo_params),
    )
    turbconv_params =
        create_parameter_struct(CAP.TurbulenceConvectionParameters;)
    return create_parameter_struct(
        CAP.ClimaAtmosParameters;
        subparam_structs = (;
            thermodynamics_params = thermo_params,
            rrtmgp_params,
            insolation_params,
            microphysics_params = microphys_params,
            surface_fluxes_params = surf_flux_params,
            turbconv_params,
        ),
    )
end
