using L2O
using Arrow
using Test
using DelimitedFiles
using JuMP, HiGHS
import ParametricOptInterface as POI

function testdataset_gen(path)
    @testset "Dataset generation: $filetype" for filetype in [CSVFile, ArrowFile]
        model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
        @variable(model, x)
        p = @variable(model, _p in POI.Parameter(1.0))
        @constraint(model, cons, x + _p >= 3)
        @objective(model, Min, 2x)

        num_p = 10
        problem_iterator = ProblemIterator(collect(1:num_p), Dict(p => collect(1.0:num_p)))
        file = joinpath(path, "test.$(string(filetype))")
        recorder = Recorder{filetype}(file, primal_variables=[:x], dual_variables=[:cons])
        solve_batch(model, problem_iterator, recorder)
        if filetype == CSVFile
            file1 = joinpath(path, "test.csv")
            @test isfile(file1)
            @test length(readdlm(file1, ',')[:, 1]) == num_p+1
            @test length(readdlm(file1, ',')[1, :]) == 3
            rm(file1)
        else
            file2 = joinpath(path, "test.arrow")
            @test isfile(file2)
            df = Arrow.Table(file2)
            @test length(df) == 3
            @test length(df[1]) == num_p
        end
    end
end

@testset "L2O.jl" begin
    mktempdir() do path
        testdataset_gen(path)
    end
end
