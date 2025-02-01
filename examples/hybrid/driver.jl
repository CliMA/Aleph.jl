redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import Random
Random.seed!(1234)

if !(@isdefined config)
    (; config_file, job_id) = CA.commandline_kwargs()
    config = CA.AtmosConfig(config_file; job_id)
end
simulation = CA.get_simulation(config)
sol_res = CA.solve_atmos!(simulation)

if ClimaComms.iamroot(config.comms_ctx)
    make_plots(Val(Symbol(reference_job_id)), simulation.output_dir)
end
