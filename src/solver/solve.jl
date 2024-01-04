import ClimaTimeSteppers as CTS
import Base.Sys: maxrss

struct EfficiencyStats{TS <: Tuple, WT}
    tspan::TS
    walltime::WT
end

mutable struct GracefulExit
    path::String
    function GracefulExit(; path)
        mkpath(path)
        file = joinpath(path, "graceful_exit.dat")
        open(file, "w") do io
            println(io, "#      To gracefully exit job associated")
            println(io, "#      with the output path:")
            println(io, "#           $path")
            println(io, "#      please change the following number from")
            println(io, "#      0 (continue running) to 1 (gracefully exit)")
            println(io, "0")
        end
        return new(path)
    end
end

simulated_years_per_day(es::EfficiencyStats) =
    simulated_years(es) / walltime_in_days(es)

simulated_years(es::EfficiencyStats) =
    (es.tspan[2] - es.tspan[1]) * (1 / (365 * 24 * 3600)) #=seconds * years per second=#
walltime_in_days(es::EfficiencyStats) = es.walltime * (1 / (24 * 3600)) #=seconds * days per second=#

function timed_solve!(integrator, graceful_exit::GracefulExit)
    walltime = @elapsed begin
        s = @timed_str begin
            _solve!(integrator, graceful_exit)
        end
    end
    @info "solve!: $s"
    es = EfficiencyStats(integrator.sol.prob.tspan, walltime)
    @info "sypd: $(simulated_years_per_day(es))"
    return (integrator.sol, walltime)
end

finalize!(integrator) = DiffEqBase.finalize!(
    integrator.callback,
    integrator.u,
    integrator.t,
    integrator,
)

function exit_step_loop(ge::GracefulExit)
    lines = readlines(joinpath(ge.path, "graceful_exit.dat"))
    exit_loop = 0
    for line in lines
        startswith(line, "#") && continue
        try
            exit_loop = parse(Int, line)
        catch
            # Let's re-set the file, in
            # case it was incorrectly edited.
            GracefulExit(; path = ge.path)
        end
        break
    end
    return exit_loop == 1
end



function _solve!(integrator, graceful_exit::GracefulExit)
    # TODO: remove assertion if upstream packages don't fail
    @assert !integrator.advance_to_tstop "Unsupported ClimaAtmos option. call step! instead of solve!"
    while !isempty(integrator.tstops) &&
              integrator.step != integrator.stepstop &&
              !exit_step_loop(graceful_exit)
        SciMLBase.step!(integrator)
    end
    finalize!(integrator)
    return integrator.sol
end

struct AtmosSolveResults{S, RT, WT}
    sol::S
    ret_code::RT
    walltime::WT
end

"""
    solve_atmos!(integrator)

Call the ClimaTimeSteppers solve on the integrator.
Returns a `AtmosSolveResults`, containing the
solution, walltime, and a return code `Symbol`
indicating one of:

 - `:success`
 - `:success`
 - `:simulation_crashed`

`try-catch` is used to allow plotting
results for simulations that have crashed.
"""
function solve_atmos!(simulation::AtmosSimulation)
    (; integrator, output_writers, output_dir, graceful_exit) = simulation
    (; tspan) = integrator.sol.prob
    @info "Running" job_id = simulation.job_id output_dir = output_dir tspan
    comms_ctx = ClimaComms.context(axes(integrator.u.c))
    SciMLBase.step!(integrator)
    precompile_callbacks(integrator)
    GC.gc()
    try
        if CA.is_distributed(comms_ctx)
            # GC.enable(false) # disabling GC causes a memory leak
            ClimaComms.barrier(comms_ctx)
            (sol, walltime) = timed_solve!(integrator, graceful_exit)
            ClimaComms.barrier(comms_ctx)
            GC.enable(true)
            return AtmosSolveResults(sol, :success, walltime)
        else
            (sol, walltime) = timed_solve!(integrator, graceful_exit)
            return AtmosSolveResults(sol, :success, walltime)
        end
    catch ret_code
        if !CA.is_distributed(comms_ctx)
            # We can only save when not distributed because we don't have a way to sync the
            # MPI processes (maybe just one MPI rank crashes, leading to a hanging
            # simulation)
            CA.save_state_to_disk_func(integrator, simulation.output_dir)
        end
        @error "ClimaAtmos simulation crashed. Stacktrace for failed simulation" exception =
            (ret_code, catch_backtrace())
        return AtmosSolveResults(nothing, :simulation_crashed, nothing)
    finally
        # Close all the files opened by the writers

        maxrss_str = prettymemory(maxrss())
        @info "Memory currently used (after solve!) by the process (RSS): $maxrss_str"

        foreach(CAD.close, output_writers)
    end
end

"""
    benchmark_step!(integrator)

# Example

To instantiate the environment:
```
julia --project=examples
```

Then, to run interactively:
```julia
import ClimaAtmos as CA
import Random
Random.seed!(1234)
config = CA.AtmosCoveragePerfConfig();
simulation = CA.get_simulation(config);
(; integrator) = simulation
Y₀ = deepcopy(integrator.u);
CA.benchmark_step!(integrator, Y₀);
```

Alternatively, this can be run non-interactively,
with adjusted flags as:
```
julia --project=examples perf/benchmark_step.jl --h_elem 6
```

See [`argparse_settings`](@ref) for the
method defining the flags, and

    https://clima.github.io/ClimaAtmos.jl/dev/cl_args/

for the flags outlined in a table.

"""
function benchmark_step!(integrator, Y₀, n_steps = 10)
    for i in 1:n_steps
        SciMLBase.step!(integrator)
        integrator.u .= Y₀ # temporary hack to simplify performance benchmark.
    end
    return nothing
end

"""
    cycle!(integrator; n_cycles = 1)

Run `step!` the least common multiple times
for all callbacks. i.e., all callbacks will
have been called at least `n_cycles` times.

`cycle!` is the true atomic unit for performance,
because, unlike `step!`, it takes all callbacks
into account.
"""
function cycle!(integrator; n_cycles = 1)
    n_steps = n_steps_per_cycle(integrator) * n_cycles
    for i in 1:n_steps
        SciMLBase.step!(integrator)
    end
    return nothing
end

function call_all_callbacks!(integrator)
    for cb! in atmos_callbacks(integrator.callback)
        cb!(integrator)
    end
    return nothing
end

"""
    precompile_atmos(integrator)

Precompiles `step!` and all callbacks
in the `integrator`.
"""
function precompile_atmos(integrator)
    B = Base.precompile(SciMLBase.step!, (typeof(integrator),))
    @assert B
    precompile_callbacks(integrator)
    return nothing
end

function precompile_callbacks(integrator)
    B = Base.precompile(call_all_callbacks!, (typeof(integrator),))
    @assert B
    return nothing
end
