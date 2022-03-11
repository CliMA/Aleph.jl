abstract type AbstractCallback end

struct Info <: AbstractCallback end
struct CFL <: AbstractCallback end

Base.@kwdef struct StateCheck{𝒜} <: AbstractCallback
    number_of_checks::𝒜
end

Base.@kwdef struct VTKState{𝒜,ℬ,𝒞,𝒟} <: AbstractCallback
    iteration::𝒜 = 1
    filepath::ℬ = "."
    counter::𝒞 = [0]
    overwrite::𝒟 = true
end

Base.@kwdef struct JLD2State{𝒜,ℬ,𝒞} <: AbstractCallback
    iteration::𝒜
    filepath::ℬ
    overwrite::𝒞 = true
end

Base.@kwdef struct PositivityPreservingCallback{𝒜} <: AbstractCallback
    filterstates::𝒜 = 6:6
end

Base.@kwdef struct ReferenceStateUpdate{𝒜} <: AbstractCallback
    recompute::𝒜 = 20
end


Base.@kwdef struct AveragedState{𝒜,ℬ,𝒞,𝒟} <: AbstractCallback
    iteration::𝒜
    filepath::ℬ
    overwrite::𝒞 = true
    start_iteration::𝒟 = 0
end

Base.@kwdef struct LatLonDiagnostics{𝒜,ℬ,𝒞,𝒟,ℰ} <: AbstractCallback
    iteration::𝒜
    filepath::ℬ
    overwrite::𝒞 = true
    start_iteration::𝒟 = 0
    latitude::ℰ
    longitude::ℰ
    radius::ℰ
end

Base.@kwdef struct BoxDiagnostics{𝒜,ℬ,𝒞,𝒟,ℰ} <: AbstractCallback
    iteration::𝒜
    filepath::ℬ
    overwrite::𝒞 = true
    start_iteration::𝒟 = 0
    x::ℰ
    y::ℰ
    z::ℰ
end

function create_callbacks(simulation::Simulation, ode_solver)
    callbacks = simulation.callbacks

    if isempty(callbacks)
        return ()
    else
        cbvector = [
            create_callback(callback, simulation, ode_solver)
            for callback in callbacks
        ]
        return tuple(cbvector...)
    end
end