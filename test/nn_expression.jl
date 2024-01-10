"""
    test_flux_jump_basic()

Tests running a jump model with a flux expression.
"""
function test_flux_jump_basic()
    for i in 1:10
        model = JuMP.Model(HiGHS.Optimizer)

        @variable(model, x[i = 1:3]>= 2.3)

        flux_model = Chain(Dense(3, 3, relu), Dense(3, 1))

        ex = flux_model(x)[1]

        # @constraint(model, ex >= -100.0)
        @constraint(model, sum(x) <= 10)

        @objective(model, Min, ex)

        JuMP.optimize!(model)

        @test termination_status(model) === OPTIMAL
        if flux_model(value.(x))[1] <= 1.0
            @test isapprox(flux_model(value.(x))[1], value(ex); atol=0.01)
        else
            @test isapprox(flux_model(value.(x))[1], value(ex); rtol=0.001)
        end
    end
end

"""
    test_FullyConnected_jump()

Tests running a jump model with a FullyConnected Network expression.
"""
function test_FullyConnected_jump()
    for i in 1:10
        X = rand(100, 3)
        Y = rand(100, 1)

        nn = MultitargetNeuralNetworkRegressor(;
            builder=FullyConnectedBuilder([8, 8, 8]),
            rng=123,
            epochs=100,
            optimiser=optimiser,
            acceleration=CUDALibs(),
            batch_size=32,
        )

        mach = machine(nn, X, y)
        fit!(mach; verbosity=2)

        flux_model = mach.fitresult[1]

        model = JuMP.Model(Gurobi.Optimizer)

        @variable(model, x[i = 1:3]>= 2.3)

        ex = flux_model(x)[1]

        # @constraint(model, ex >= -100.0)
        @constraint(model, sum(x) <= 10)

        @objective(model, Min, ex)

        JuMP.optimize!(model)

        @test termination_status(model) === OPTIMAL
        if flux_model(value.(x))[1] <= 1.0
            @test isapprox(flux_model(value.(x))[1], value(ex); atol=0.01)
        else
            @test isapprox(flux_model(value.(x))[1], value(ex); rtol=0.001)
        end
    end
end

function print_conflict!(model)
    JuMP.compute_conflict!(model)
    ctypes = list_of_constraint_types(model)
    for (F, S) in ctypes
        cons = all_constraints(model, F, S)
        for i in eachindex(cons)
            isassigned(cons, i) || continue
            con = cons[i]
            cst = MOI.get(model, MOI.ConstraintConflictStatus(), con)
            cst == MOI.IN_CONFLICT && @info JuMP.name(con) con
        end
    end
    return nothing
end
