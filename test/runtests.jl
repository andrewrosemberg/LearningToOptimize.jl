using L2O
using Arrow
using Test
using DelimitedFiles
using JuMP, HiGHS
import ParametricOptInterface as POI

"""
    testdataset_gen(path::String)

Test dataset generation for different filetypes
"""
function testdataset_gen(path::String)
    @testset "Type: $filetype" for filetype in [CSVFile, ArrowFile]
        # The problem to iterate over
        model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
        @variable(model, x)
        p = @variable(model, _p in POI.Parameter(1.0))
        @constraint(model, cons, x + _p >= 3)
        @objective(model, Min, 2x)

        # The problem iterator
        num_p = 10
        problem_iterator = ProblemIterator(collect(1:num_p), Dict(p => collect(1.0:num_p)))

        # The recorder
        file = joinpath(path, "test.$(string(filetype))") # file path
        recorder = Recorder{filetype}(file; primal_variables=[:x], dual_variables=[:cons])

        # Solve all problems and record solutions
        solve_batch(model, problem_iterator, recorder)

        # Check if file exists and has the correct number of rows and columns
        if filetype == CSVFile
            file1 = joinpath(path, "test.csv")
            @test isfile(file1)
            @test length(readdlm(file1, ',')[:, 1]) == num_p + 1
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
    @testset "Dataset Generation" begin
        mktempdir() do path
            # Different filetypes
            testdataset_gen(path)
            # Pglib
            @testset "pg_lib case" begin
                include(
                    joinpath(
                        dirname(dirname(@__FILE__)), "examples", "powermodels", "pg_lib.jl"
                    ),
                )
            end
        end
    end
end
