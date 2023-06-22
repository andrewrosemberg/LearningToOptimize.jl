using L2O
using Test
using JuMP, HiGHS
import ParametricOptInterface as POI

@testset "L2O.jl" begin
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    @variable(model, x)
    p = @variable(model, _p in POI.Parameter(1.0))
    @constraint(model, cons, x + _p >= 3)
    @objective(model, Min, 2x)

    MOI.set(model, POI.ParameterValue(), p, 2.0)
    optimize!(model)
end
