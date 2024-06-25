using Test
import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
import ClimaAtmos as CA
import SciMLBase as SMB

testfun!() = π
cb_default = CA.call_every_n_steps(testfun!;)
test_nsteps = 999
test_dt = 1
test_tend = 999.0

cb_1 = CA.call_every_n_steps(
    testfun!,
    test_nsteps;
    skip_first = false,
    call_at_end = false,
    condition = nothing,
)
cb_2 =
    CA.call_every_dt(testfun!, test_dt; skip_first = false, call_at_end = false)
cb_3 = CA.callback_from_affect(cb_2.affect!)
cb_4 = CA.call_every_n_steps(
    testfun!,
    3;
    skip_first = false,
    call_at_end = false,
    condition = nothing,
)
cb_set = SMB.CallbackSet(cb_1, cb_2, cb_4)

@testset "simple default callback" begin
    @test cb_default.condition.n == 1
    @test cb_default.affect!.f!() == π
end

# per n steps
@testset "every n-steps callback" begin
    @test cb_1.initialize.skip_first == false
    @test cb_1.condition.n == test_nsteps
    @test cb_1.affect!.f!() == π
    @test_throws AssertionError CA.call_every_n_steps(
        testfun!,
        Inf;
        skip_first = false,
        call_at_end = false,
    )
end

# per dt interval
@testset "dt interval callback" begin
    @test cb_2 isa SMB.DiscreteCallback
    @test cb_2.affect!.dt == test_dt
    @test cb_2.affect!.cb!.f!() == π
    @test_throws AssertionError CA.call_every_dt(
        testfun!,
        Inf;
        skip_first = false,
        call_at_end = false,
    )
end

@testset "atmos callbacks and callback sets" begin
    # atmoscallbacks from discrete callbacks
    @test cb_3.f!() == π
    atmos_cbs = CA.atmos_callbacks(cb_set)
    # test against expected callback outcomes
    tspan = [0, test_tend]
    @test CA.n_expected_calls(cb_set, test_dt, tspan)[2] == test_tend
    @test CA.n_steps_per_cycle_per_cb(cb_set, test_dt) ==
          [test_nsteps; test_dt; 3]
end

# query functions
