using StaticArrays
include("../src/interface/domains.jl")
include("../src/interface/models.jl")
include("../src/interface/physics.jl")
include("../src/interface/boundary_conditions.jl")
include("../src/interface/grids.jl")
include("../src/backends/backends.jl")
include("../src/interface/timestepper_abstractions.jl")
include("../src/backends/dg_model_backends/backend_hook.jl")
include("../src/interface/simulations.jl")
include("../src/interface/callbacks.jl")
include("../src/backends/dg_model_backends/boilerplate.jl")
include("../src/utils/sphere_utils.jl")

# to be removed
using CLIMAParameters #: AbstractEarthParameterSet
struct PlanetParameterSet <: AbstractEarthParameterSet end
get_planet_parameter(p::Symbol) = getproperty(CLIMAParameters.Planet, p)(PlanetParameterSet())

# set up shadyCFL: helper function
function shady_timestep(discretized_domain::DiscretizedDomain; vcfl = 16, hcfl = 0.15, sound_speed = 330)
    # vertical cfl
    domain = discretized_domain.domain
    height = domain.height
    ne = discretized_domain.discretization.vertical.elements
    np = discretized_domain.discretization.vertical.polynomial_order
    vdt = height / ne / (np^2 + 1) / sound_speed * vcfl
    @info "vertical cfl implies dt=$vdt"
    # horizontal cfl
    circumference = domain.radius * 2π
    ne = discretized_domain.discretization.horizontal.elements * 4 # since 4 faces on cubed sphere equator
    np = discretized_domain.discretization.horizontal.polynomial_order
    hdt = circumference / ne / (np^2 + 1) / sound_speed * hcfl
    @info "horizontal cfl implies dt=$hdt"

    if vdt < hdt 
        dt = vdt
        @info "limited by vertical acoustic modes dt=$dt seconds"
    else
        dt = hdt
        @info "limited by horizontal acoustic modes dt=$dt seconds"

    end

    return dt
end

function legit_timestep(discretized_domain::DiscretizedDomain, backend::AbstractBackend; vcfl = 16.0, hcfl = 0.4, sound_speed = 330)
    numerical_grid = create_grid(backend, discretized_domain);
    cₛ = sound_speed
    Δxᵥ = min_node_distance(numerical_grid, VerticalDirection())
    Δxₕ = min_node_distance(numerical_grid, HorizontalDirection()) 
    vdt =  (Δxᵥ / cₛ) * vcfl
    hdt =  (Δxₕ / cₛ) * hcfl


    println("The vertical minimum grid spacing is ", Δxᵥ , " meters" )
    println("The horizontal minimum grid spacing is ", Δxₕ / 1e3 , " kilometers")
    println("The vertical CFL is ", vcfl)
    println("The horizontal CFL is ", hcfl)

    if vdt < hdt
        dt = vdt
        @info "limited by vertical acoustic modes dt=$dt seconds"
    else
        dt = hdt
        @info "limited by horizontal acoustic modes dt=$dt seconds"

    end
    return dt
end

# create jld2 name: helper function
function create_jld2_name(base_name, discretized_domain, numerical_flux)
    he = string(discretized_domain.discretization.horizontal.elements)
    hp = string(discretized_domain.discretization.horizontal.polynomial_order)
    ve = string(discretized_domain.discretization.vertical.elements)
    vp = string(discretized_domain.discretization.vertical.polynomial_order)
    return base_name * "_" * "he_" * he * "_" * "hp_" * hp * "_" * "ve_" * ve * "_" * "vp_" * vp * "_" * string(numerical_flux) * ".jld2"
end

parameters = (
    a    = get_planet_parameter(:planet_radius),
    Ω    = get_planet_parameter(:Omega),
    g    = get_planet_parameter(:grav),
    κ    = get_planet_parameter(:kappa_d),
    R_d  = get_planet_parameter(:R_d),
    cv_d = get_planet_parameter(:cv_d),
    cp_d = get_planet_parameter(:cp_d),
    γ    = get_planet_parameter(:cp_d)/get_planet_parameter(:cv_d),
    H    = 30e3,
    pₒ   = 1.0e5,
    k    = 3.0,
    Γ    = 0.005,
    T_0  = 0.0,
    T_E  = 310.0,
    T_P  = 240.0,
    b    = 2.0,
    z_t  = 15e3,
    λ_c  = π / 9,
    ϕ_c  = 2 * π / 9,
    V_p  = 1.0,
    day = 86400,
    p0 = 1e5,
    T_ref = 255,
)

# Held-Suarez forcing
struct HeldSuarezForcing{S} <: AbstractForcing
    parameters::S
end

FT = Float64
day = 86400
held_suarez_parameters = (;
    k_a = FT(1 / (40 * day)),
    k_f = FT(1 / day),
    k_s = FT(1 / (4 * day)),
    ΔT_y = FT(60),
    Δθ_z = FT(10),
    T_equator = FT(315),
    T_min = FT(200),
    σ_b = FT(7 / 10),
    R_d  = parameters.R_d,
    day  = parameters.day,
    grav = parameters.g,
    cp_d = parameters.cp_d,
    cv_d = parameters.cv_d,
    MSLP = parameters.p0,
)

function calc_source!(
    source,
    balance_law::ThreeDimensionalDryCompressibleEulerWithTotalEnergy,
    hsf::HeldSuarezForcing,
    state,
    aux,
)

    FT = eltype(state)

    _R_d  = hsf.parameters.R_d
    _day  = hsf.parameters.day
    _grav = hsf.parameters.grav
    _cp_d = hsf.parameters.cp_d
    _cv_d = hsf.parameters.cv_d
    _p0   = hsf.parameters.MSLP

    # Parameters
    T_ref = FT(255)

    # Extract the state
    ρ = state.ρ
    ρu = state.ρu
    ρe = state.ρe
    Φ = aux.Φ

    x = aux.x
    y = aux.y
    z = aux.z
    coord = @SVector[x,y,z]

    p = calc_pressure(balance_law.equation_of_state, state, aux, balance_law.parameters)
    T = p / (ρ * _R_d)

    # Held-Suarez parameters
    k_a  = hsf.parameters.k_a
    k_f  = hsf.parameters.k_f
    k_s  = hsf.parameters.k_s
    ΔT_y = hsf.parameters.ΔT_y
    Δθ_z = hsf.parameters.Δθ_z
    T_equator = hsf.parameters.T_equator
    T_min = hsf.parameters.T_min
    σ_b = hsf.parameters.σ_b

    # Held-Suarez forcing
    φ = @inbounds asin(coord[3] / norm(coord, 2))

    #TODO: replace _p0 with dynamic surfce pressure in Δσ calculations to account
    #for topography, but leave unchanged for calculations of σ involved in T_equil
    σ = p / _p0
    exner_p = σ^(_R_d / _cp_d)
    Δσ = (σ - σ_b) / (1 - σ_b)
    height_factor = max(0, Δσ)
    T_equil = (T_equator - ΔT_y * sin(φ) * sin(φ) - Δθ_z * log(σ) * cos(φ) * cos(φ)) * exner_p
    T_equil = max(T_min, T_equil)

    k_T = k_a + (k_s - k_a) * height_factor * cos(φ) * cos(φ) * cos(φ) * cos(φ) 
    k_v = k_f * height_factor

    # horizontal projection
    k = coord / norm(coord)
    P = I - k * k'

    # Apply Held-Suarez forcing
    source.ρu -= k_v * P * ρu

    source.ρe -= k_T * ρ * _cv_d * (T - T_equil)

    return nothing
end

# set up initial condition
# additional initial condition parameters
T₀(𝒫)   = 0.5 * (𝒫.T_E + 𝒫.T_P)
A(𝒫)    = 1.0 / 𝒫.Γ
B(𝒫)    = (T₀(𝒫) - 𝒫.T_P) / T₀(𝒫) / 𝒫.T_P
C(𝒫)    = 0.5 * (𝒫.k + 2) * (𝒫.T_E - 𝒫.T_P) / 𝒫.T_E / 𝒫.T_P
H(𝒫)    = 𝒫.R_d * T₀(𝒫) / 𝒫.g
d_0(𝒫)  = 𝒫.a / 6

# convenience functions that only depend on height
τ_z_1(𝒫,r)   = exp(𝒫.Γ * (r - 𝒫.a) / T₀(𝒫))
τ_z_2(𝒫,r)   = 1 - 2 * ((r - 𝒫.a) / 𝒫.b / H(𝒫))^2
τ_z_3(𝒫,r)   = exp(-((r - 𝒫.a) / 𝒫.b / H(𝒫))^2)
τ_1(𝒫,r)     = 1 / T₀(𝒫) * τ_z_1(𝒫,r) + B(𝒫) * τ_z_2(𝒫,r) * τ_z_3(𝒫,r)
τ_2(𝒫,r)     = C(𝒫) * τ_z_2(𝒫,r) * τ_z_3(𝒫,r)
τ_int_1(𝒫,r) = A(𝒫) * (τ_z_1(𝒫,r) - 1) + B(𝒫) * (r - 𝒫.a) * τ_z_3(𝒫,r)
τ_int_2(𝒫,r) = C(𝒫) * (r - 𝒫.a) * τ_z_3(𝒫,r)
F_z(𝒫,r)     = (1 - 3 * ((r - 𝒫.a) / 𝒫.z_t)^2 + 2 * ((r - 𝒫.a) / 𝒫.z_t)^3) * ((r - 𝒫.a) ≤ 𝒫.z_t)

# convenience functions that only depend on longitude and latitude
d(𝒫,λ,ϕ)     = 𝒫.a * acos(sin(ϕ) * sin(𝒫.ϕ_c) + cos(ϕ) * cos(𝒫.ϕ_c) * cos(λ - 𝒫.λ_c))
c3(𝒫,λ,ϕ)    = cos(π * d(𝒫,λ,ϕ) / 2 / d_0(𝒫))^3
s1(𝒫,λ,ϕ)    = sin(π * d(𝒫,λ,ϕ) / 2 / d_0(𝒫))
cond(𝒫,λ,ϕ)  = (0 < d(𝒫,λ,ϕ) < d_0(𝒫)) * (d(𝒫,λ,ϕ) != 𝒫.a * π)

# base-state thermodynamic variables
I_T(𝒫,ϕ,r)   = (cos(ϕ) * r / 𝒫.a)^𝒫.k - 𝒫.k / (𝒫.k + 2) * (cos(ϕ) * r / 𝒫.a)^(𝒫.k + 2)
T(𝒫,ϕ,r)     = (τ_1(𝒫,r) - τ_2(𝒫,r) * I_T(𝒫,ϕ,r))^(-1) * (𝒫.a/r)^2
p(𝒫,ϕ,r)     = 𝒫.pₒ * exp(-𝒫.g / 𝒫.R_d * (τ_int_1(𝒫,r) - τ_int_2(𝒫,r) * I_T(𝒫,ϕ,r)))
θ(𝒫,ϕ,r)     = T(𝒫,ϕ,r) * (𝒫.pₒ / p(𝒫,ϕ,r))^𝒫.κ

# base-state velocity variables
U(𝒫,ϕ,r)  = 𝒫.g * 𝒫.k / 𝒫.a * τ_int_2(𝒫,r) * T(𝒫,ϕ,r) * ((cos(ϕ) * r / 𝒫.a)^(𝒫.k - 1) - (cos(ϕ) * r / 𝒫.a)^(𝒫.k + 1))
u(𝒫,ϕ,r)  = -𝒫.Ω * r * cos(ϕ) + sqrt((𝒫.Ω * r * cos(ϕ))^2 + r * cos(ϕ) * U(𝒫,ϕ,r))
v(𝒫,ϕ,r)  = 0.0
w(𝒫,ϕ,r)  = 0.0

# velocity perturbations
δu(𝒫,λ,ϕ,r)  = -16 * 𝒫.V_p / 3 / sqrt(3) * F_z(𝒫,r) * c3(𝒫,λ,ϕ) * s1(𝒫,λ,ϕ) * (-sin(𝒫.ϕ_c) * cos(ϕ) + cos(𝒫.ϕ_c) * sin(ϕ) * cos(λ - 𝒫.λ_c)) / sin(d(𝒫,λ,ϕ) / 𝒫.a) * cond(𝒫,λ,ϕ)
δv(𝒫,λ,ϕ,r)  = 16 * 𝒫.V_p / 3 / sqrt(3) * F_z(𝒫,r) * c3(𝒫,λ,ϕ) * s1(𝒫,λ,ϕ) * cos(𝒫.ϕ_c) * sin(λ - 𝒫.λ_c) / sin(d(𝒫,λ,ϕ) / 𝒫.a) * cond(𝒫,λ,ϕ)
δw(𝒫,λ,ϕ,r)  = 0.0

# CliMA prognostic variables
# compute the total energy
uˡᵒⁿ(𝒫,λ,ϕ,r)   = u(𝒫,ϕ,r) + δu(𝒫,λ,ϕ,r)
uˡᵃᵗ(𝒫,λ,ϕ,r)   = v(𝒫,ϕ,r) + δv(𝒫,λ,ϕ,r)
uʳᵃᵈ(𝒫,λ,ϕ,r)   = w(𝒫,ϕ,r) + δw(𝒫,λ,ϕ,r)

e_int(𝒫,λ,ϕ,r)  = (𝒫.R_d / 𝒫.κ - 𝒫.R_d) * (T(𝒫,ϕ,r) - 𝒫.T_0)
e_kin(𝒫,λ,ϕ,r)  = 0.5 * ( uˡᵒⁿ(𝒫,λ,ϕ,r)^2 + uˡᵃᵗ(𝒫,λ,ϕ,r)^2 + uʳᵃᵈ(𝒫,λ,ϕ,r)^2 )
e_pot(𝒫,λ,ϕ,r)  = 𝒫.g * r

ρ₀(𝒫,λ,ϕ,r)    = p(𝒫,ϕ,r) / 𝒫.R_d / T(𝒫,ϕ,r)
ρuˡᵒⁿ(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * uˡᵒⁿ(𝒫,λ,ϕ,r)
ρuˡᵃᵗ(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * uˡᵃᵗ(𝒫,λ,ϕ,r)
ρuʳᵃᵈ(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * uʳᵃᵈ(𝒫,λ,ϕ,r)

ρe(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * (e_int(𝒫,λ,ϕ,r) + e_kin(𝒫,λ,ϕ,r) + e_pot(𝒫,λ,ϕ,r))

# Cartesian Representation (boiler plate really)
ρ₀ᶜᵃʳᵗ(𝒫, x...)  = ρ₀(𝒫, lon(x...), lat(x...), rad(x...))
ρu₀ᶜᵃʳᵗ(𝒫, x...) = (   ρuʳᵃᵈ(𝒫, lon(x...), lat(x...), rad(x...)) * r̂(x...)
                     + ρuˡᵃᵗ(𝒫, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...)
                     + ρuˡᵒⁿ(𝒫, lon(x...), lat(x...), rad(x...)) * λ̂(x...) )
ρe₀ᶜᵃʳᵗ(𝒫, x...) = ρe(𝒫, lon(x...), lat(x...), rad(x...))


# set up reference state
ref_state = DryReferenceState(DecayingTemperatureProfile{FT}(parameters, FT(290), FT(220), FT(8e3)))

# Set up model
model = ModelSetup(
    equations = ThreeDimensionalEuler(
        thermodynamic_variable = TotalEnergy(),
        equation_of_state = DryIdealGas(),
        pressure_convention = Compressible(),
        sources = (
            coriolis = DeepShellCoriolis(),
            gravity = Gravity(),
            forcing = HeldSuarezForcing(held_suarez_parameters),
        ),
        ref_state = ref_state,
    ),
    boundary_conditions = (DefaultBC(), DefaultBC()),
    # boundary_conditions = (
    #     ρ  = (top = NoFlux(), bottom = NoFlux(),),
    #     ρu = (top = FreeSlip(), bottom = FreeSlip(),),
    #     ρe = (top = NoFlux(), bottom = NoFlux(),),
    # ),
    initial_conditions = (
        ρ = ρ₀ᶜᵃʳᵗ, ρu = ρu₀ᶜᵃʳᵗ, ρe = ρe₀ᶜᵃʳᵗ,
    ),
    parameters = parameters,
)

flux = :refanov
 he = 15
 hp = 3
 ve = 15
 vp = 3
 jld_name = "test_hs"
  sim_days = 0.01
  dt = nothing
  recompute_minutes = 30
  vcfl = 1.0
  hcfl = 0.15
  sound_speed = 353
    parameters = model.parameters
    # set up backend
    backend = DiscontinuousGalerkinBackend(numerics = (flux = flux,),)

    # Set up grid
    domain = SphericalShell(
        radius = parameters.a,
        height = parameters.H,
    )

    discretized_domain = DiscretizedDomain(
        domain = domain,
        discretization = (
            horizontal = SpectralElementGrid(elements = he, polynomial_order = hp),
            vertical = SpectralElementGrid(elements = ve, polynomial_order = vp)
        ),
    )

    if dt == nothing
        # dt = shady_timestep(discretized_domain)
        # println("shady dt = ", dt)
        dt = legit_timestep(discretized_domain, backend, vcfl = vcfl, hcfl = hcfl, sound_speed = sound_speed)
        dt = minimum([dt, 60.0])
        println("legit dt = ", dt)
    else
        println("dt = ", dt)
    end
    recompute = floor(Int, recompute_minutes * 60 / dt) # recompute fields at every 30 minutes
    println("recomputing at ", recompute)

    # CHANGE DEFAULT SAVING STUFF
    jld_it = floor(Int, 50 * 60 * 60 / dt) # floor(Int, 50 * 24 * 60 * 60 / dt) # every 50 days
    jld_filepath = create_jld2_name(jld_name, discretized_domain, flux)

    avg_start = floor(Int, 200 * 24 * 60 * 60 / dt) # start after 200 days
    jld_it_2  = floor(Int, 4*6 * 60 * 60 / dt)        # save average every 4*6 hours

    lat_grd = collect(-90:1.0:90) .* 1.0
    long_grd = collect(-180:1.0:180) .* 1.0
    rad_grd = collect(domain.radius:500:(domain.radius + domain.height)) .* 1.0
    
    ll_cb = LatLonDiagnostics(
    iteration = jld_it_2, 
    filepath = "avg_" * jld_filepath,
    start_iteration = avg_start,
    latitude = lat_grd,
    longitude = long_grd,
    radius = rad_grd)

    jl_cb = JLD2State(iteration = jld_it, filepath = jld_filepath)
    
    # set up simulation
    simulation = Simulation(
        backend = backend,
        discretized_domain = discretized_domain,
        model = model,
        splitting = IMEXSplitting(linear_model = :verylinear, ),
        timestepper = (
            method = IMEX(),
            start = 0.0,
            finish = sim_days *  24 * 3600,
            timestep = dt,
        ),
        callbacks = (
            Info(),
            ReferenceStateUpdate(recompute = recompute),
            # ll_cb,
            jl_cb,
        ),
    )

    # run the simulation
    initialize!(simulation)
    tic = time()
    try
        evolve!(simulation)
    catch err
        @info "evolve has thrown an error"
        showerror(stdout, err )
    end
    toc = time()
    println("The amount of time for the simulation was ", (toc - tic)/(3600), " hours")

    # Check the domain average 
    #=
    initialize!(simulation)
    numerical_grid = create_grid(backend, discretized_domain)
    M = massmatrix(numerical_grid)
    ρᴮ  = simulation.state[:,1,:]
    ρeᴮ = simulation.state[:,5,:]
    ρ̅ᴮ  = sum(M .* ρᴮ) / sum(M)
    ρ̅e̅ᴮ = sum(M .* ρeᴮ) / sum(M)

    tic = time()
    try
        evolve!(simulation)
    catch err
        @info "evolve has thrown an error"
        showerror(stdout, err )
    end
    toc = time()
    println("The amount of time for the simulation was ", (toc - tic)/(3600), " hours")

    ρᴬ  = simulation.state[:,1,:]
    ρeᴬ = simulation.state[:,5,:]
    ρ̅ᴬ  = sum(M .* ρᴬ)  / sum(M)
    ρ̅e̅ᴬ = sum(M .* ρeᴬ) / sum(M)

    println("The change in mass is ", abs(ρ̅ᴬ-ρ̅ᴮ)/ρ̅ᴬ )
    println("The change in total energy is ", abs(ρ̅e̅ᴬ-ρ̅e̅ᴮ)/ρ̅e̅ᴬ )
    =#
