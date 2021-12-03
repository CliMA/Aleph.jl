"""
    ShallowWaterModel <: AbstractModel

A shallow water model. Required fields are `domain` and `parameters`.
"""
Base.@kwdef struct ShallowWaterModel{D <: AbstractHorizontalDomain, P} <:
                   AbstractModel
    domain::D
    parameters::P
    name::Symbol = :swm
    varnames::Tuple = (:h, :u, :c)
end

function Models.default_initial_conditions(model::ShallowWaterModel)
    space = make_function_space(model.domain)
    local_geometry = Fields.local_geometry_field(space)

    # functions that make zeros for this model
    zero_val = zero(Spaces.undertype(space))
    zero_scalar(lg) = zero_val
    zero_vector(lg) = Geometry.Covariant12Vector(zero_val, zero_val)

    h0 = zero_scalar.(local_geometry)
    u0 = zero_vector.(local_geometry)
    c0 = zero_scalar.(local_geometry)

    return Fields.FieldVector(swm = Fields.FieldVector(h = h0, u = u0, c = c0))
end

function Models.make_ode_function(model::ShallowWaterModel)
    function rhs!(dY, Y, Ya, t)
        @unpack D₄, g = model.parameters

        # unpack tendencies and state
        dYm = dY.swm
        dh = dYm.h
        du = dYm.u
        dc = dYm.c
        Ym = Y.swm
        h = Ym.h
        u = Ym.u
        c = Ym.c

        # operators
        sdiv = Operators.Divergence()
        wdiv = Operators.WeakDivergence()
        grad = Operators.Gradient()
        wgrad = Operators.WeakGradient()
        curl = Operators.Curl()
        wcurl = Operators.WeakCurl()

        # compute hyperviscosity first
        @. du =
            wgrad(sdiv(u)) -
            Geometry.Covariant12Vector(wcurl(Geometry.Covariant3Vector(curl(
                u,
            ))))
        @. dc = wdiv(grad(c))
        Spaces.weighted_dss!(du)
        Spaces.weighted_dss!(dc)
        @. du =
            -D₄ * (
                wgrad(sdiv(du)) -
                Geometry.Covariant12Vector(wcurl(Geometry.Covariant3Vector(curl(
                    du,
                ))),)
            )
        @. dc = -D₄ * wdiv(grad(dc))

        # add in advection terms
        space = axes(h)
        J = Fields.Field(space.local_geometry.J, space)
        @. dh = -wdiv(h * u)
        @. du +=
            -grad(g * h + norm(u)^2 / 2) +
            Geometry.Covariant12Vector((J * (u × curl(u))))
        @. dc += -wdiv(c * u)
        Spaces.weighted_dss!(dh)
        Spaces.weighted_dss!(du)
        Spaces.weighted_dss!(dc)

        return dY
    end
end
