import ArgParse
using Test
import ClimaAtmos as CA
import ClimaComms
import ClimaCore.InputOutput as InputOutput
import ClimaCore.Fields as Fields
using LinearAlgebra: norm

function parse_commandline()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! s begin
        "--output_folder_1"
        help = "Output file for job 1"
        arg_type = String
        "--output_folder_2"
        help = "Output file for job 2"
        arg_type = String
        "--t_end"
        help = "Simulation end time. This must match the target jobs"
        arg_type = String
        "--compare_state"
        help = "A bool: compare the state between output_folder_1 and output_folder_2 [`true` [default], `false`]"
        arg_type = Bool
        default = true
    end
    parsed_args = ArgParse.parse_args(ARGS, s)
    return (s, parsed_args)
end

(s, parsed_args) = parse_commandline()
t_end = CA.time_to_seconds(parsed_args["t_end"])
day = floor(Int, t_end / (60 * 60 * 24))
sec = floor(Int, t_end % (60 * 60 * 24))

function get_data(folder, name)
    reader = InputOutput.HDF5Reader(
        joinpath(folder, "day$day.$sec.hdf5"),
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()),
    )
    return InputOutput.read_field(reader, name)
end
Y_1 = get_data(parsed_args["output_folder_1"], "Y");
Y_2 = get_data(parsed_args["output_folder_2"], "Y");

compare(a::FV, b::FV, pn0 = "") where {FV <: Fields.FieldVector} =
    compare(true, a, b, pn0)
function _compare(zero_error, a::FV, b::FV, pn0 = "") where {FV}
    for pn in propertynames(a)
        pa = getproperty(a, pn)
        pb = getproperty(b, pn)
        zero_error = compare(zero_error, pa, pb, "$pn0.$pn")
    end
    return zero_error
end

compare(zero_error, a::FV, b::FV, pn0 = "") where {FV <: Fields.FieldVector} =
    _compare(zero_error, a, b, pn0)
compare(a::F, b::F, pn0 = "") where {F <: Fields.Field} =
    compare(true, a, b, pn0)
function compare(zero_error, a::F, b::F, pn0 = "") where {F <: Fields.Field}
    isempty(propertynames(a)) || return _compare(zero_error, a, b, pn0)
    abs_err = abs.(Array(parent(a)) .- Array(parent(b)))
    rel_err =
        norm(Array(parent(a)) .- Array(parent(b))) ./ norm(Array(parent(a)))
    if !(maximum(abs_err) == 0.0) || !(maximum(rel_err) == 0.0)
        @warn "--- Failed field comparison $pn0" a b max_abs_err =
            maximum(abs_err) rel_err
        zero_error = false
    end
    return zero_error
end

@testset "Compare outputs" begin
    @info "Comparing state vectors:"
    if parsed_args["compare_state"]
        @test compare(Y_1, Y_2)
    else
        @test_broken compare(Y_1, Y_2)
    end
end
