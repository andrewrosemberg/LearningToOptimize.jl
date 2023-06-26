using L2O
using Test
using DelimitedFiles
using JuMP, HiGHS
import ParametricOptInterface as POI

@testset "L2O.jl" begin
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    @variable(model, x)
    p = @variable(model, _p in POI.Parameter(1.0))
    @constraint(model, cons, x + _p >= 3)
    @objective(model, Min, 2x)

    problem_iterator = zip(1:10, (zip(p, 1.0) for i in 1:10))
    recorder = CSVRecorder("test.csv", primal_variables=[:x], dual_variables=[:cons])
    solve_batch(model, problem_iterator, recorder)
    @test isfile("test.csv")
    @test length(readdlm("test.csv", ',')[:, 1]) == 11
    @test length(readdlm("test.csv", ',')[1, :]) == 3
    rm("test.csv")
end
