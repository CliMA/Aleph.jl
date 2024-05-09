#=
Run this script with, for example:
```
julia --project=examples perf/benchmark_step.jl --h_elem 6
```
Or, interactively,
```
julia --project=examples
push!(ARGS, "--h_elem", "6")
# push!(ARGS, "--device", "CPUSingleThreaded") # uncomment to run on CPU
include(joinpath("perf", "benchmark_step.jl"));
```
=#
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import Random
Random.seed!(1234)
import ClimaAtmos as CA
import ClimaComms

include("common.jl")
(; config_file, job_id) = CA.commandline_kwargs()
config = CA.AtmosConfig((default_perf_config_file, config_file); job_id)

simulation = CA.get_simulation(config)
(; integrator) = simulation;
Y₀ = deepcopy(integrator.u);
@info "Compiling benchmark_step!..."
CA.benchmark_step!(integrator, Y₀); # compile first

@info "Running benchmark_step!..."
n_steps = 10
comms_ctx = ClimaComms.context(integrator.u.c)
device = ClimaComms.device(comms_ctx)
e = ClimaComms.@elapsed device begin
    s = CA.@timed_str begin
        CA.benchmark_step!(integrator, Y₀, n_steps) # run
    end
end
@info "Ran step! $n_steps times in $s, ($(CA.prettytime(e/n_steps*1e9)) per step)"
