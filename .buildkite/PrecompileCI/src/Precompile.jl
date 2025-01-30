module PrecompileCI

using PrecompileTools
import ClimaAtmos as CA
import ClimaComms
import ClimaCore: InputOutput, Meshes, Spaces, Quadratures
import ClimaParams

@compile_workload begin
    FT = Float32
    h_elem = 6
    z_elem = 10
    z_max = FT(30000.0)
    dz_bottom = FT(500)
    z_stretch = Meshes.HyperbolicTangentStretching(dz_bottom)
    bubble = true
    topography = "NoWarp"

    quad = Quadratures.GLL{4}()
    params = CA.ClimaAtmosParameters(FT)
    radius = CA.Parameters.planet_radius(params)
    comms_ctx = ClimaComms.context()

    horz_mesh = CA.cubed_sphere_mesh(; radius, h_elem)
    h_space = CA.make_horizontal_space(horz_mesh, quad, comms_ctx, bubble)
    CA.make_hybrid_spaces(h_space, z_max, z_elem, z_stretch; parsed_args = Dict("topography" => topography))
end


end # module Precompile
