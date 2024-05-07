redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import Random
Random.seed!(1234)
import ClimaAtmos as CA
import ClimaComms
using CairoMakie
using PrettyTables
import YAML

# Need to generate config_dict here to override `h_elem` in the loop below
parsed_args = CA.parse_commandline(CA.argparse_settings())
config_file = parsed_args["config_file"]
config_dict = YAML.load_file(config_file)
config_id = CA.config_id_from_config_file(config_file)
output_dir = config_id

steptimes = []
# Iterate through varying number of horizontal elements
for h_elem in 8:8:40
    config_dict["h_elem"] = h_elem
    config = CA.AtmosConfig(config_dict)
    simulation = CA.get_simulation(config)
    (; integrator) = simulation
    Y₀ = deepcopy(integrator.u)

    @info "Compiling benchmark_step for h_elem=$h_elem"
    CA.benchmark_step!(integrator, Y₀) # compile first

    @info "Running benchmark_step for h_elem=$h_elem"
    n_steps = 10
    comms_ctx = ClimaComms.context(integrator.u.c)
    device = ClimaComms.device(comms_ctx)
    e = ClimaComms.@elapsed device begin
        s = CA.@timed_str begin
            CA.benchmark_step!(integrator, Y₀, n_steps) # run
        end
    end
    @info "Ran step! $n_steps times in $s, ($(CA.prettytime(e/n_steps*1e9)) per step)"
    steptime = e / n_steps * 1e9
    push!(steptimes, (h_elem, steptime))
end

# Output a table with step times
data = hcat(first.(steptimes), CA.prettytime.(last.(steptimes)))
pretty_table(
    data;
    title = "Step times v/s horizontal elements",
    header = ["h_elem", "step time"],
    alignment = :l,
    crop = :none,
)

# Output a plot of step time scaling
fig = Figure(; size = (500, 500), fontsize = 40)
generic_axis = fig[1, 1] = GridLayout()
title = "Step Times v/s Horizontal Elements"
xlabel = "h_elem"
ylabel = "time (ms)"
label = "step time"
axis = Axis(generic_axis[1, 1]; title, xlabel, ylabel, yscale = identity)
CairoMakie.lines!(
    first.(steptimes),
    last.(steptimes) ./ 1e9 .* 1e3;
    title,
    label,
    linestyle = :solid,
)
CairoMakie.save(joinpath(output_dir, "scaling.png"), fig)
