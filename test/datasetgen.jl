
"""
    testdataset_gen(path::AbstractString)

Test dataset generation for different filetypes
"""
function testdataset_gen(path::AbstractString)
    @testset "Dataset Generation Type: $filetype" for filetype in [CSVFile, ArrowFile]
        # The problem to iterate over
        model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
        @variable(model, x)
        p = @variable(model, _p in POI.Parameter(1.0))
        @constraint(model, cons, x + _p >= 3)
        @objective(model, Min, 2x)

        # The problem iterator
        num_p = 10
        @test_throws AssertionError ProblemIterator(
            collect(1:num_p), Dict(p => collect(1.0:3.0))
        )
        @test_throws MethodError ProblemIterator(
            collect(1.0:3.0), Dict(p => collect(1.0:3.0))
        )
        problem_iterator = ProblemIterator(collect(1:num_p), Dict(p => collect(1.0:num_p)))
        file_input = joinpath(path, "test_input.$(string(filetype))") # file path
        save(problem_iterator, file_input, filetype)
        @test isfile(file_input)

        # The recorder
        file_output = joinpath(path, "test_output.$(string(filetype))") # file path
        @test Recorder{filetype}(file_output; primal_variables=[x]) isa Recorder{filetype}
        @test Recorder{filetype}(file_output; dual_variables=[cons]) isa Recorder{filetype}
        recorder = Recorder{filetype}(
            file_output; primal_variables=[x], dual_variables=[cons]
        )

        # Solve all problems and record solutions
        solve_batch(model, problem_iterator, recorder)

        # Check if file exists and has the correct number of rows and columns
        @test isfile(file_output)
        if filetype == CSVFile
            # test input file
            @test length(readdlm(file_input, ',')[:, 1]) == num_p + 1
            @test length(readdlm(file_input, ',')[1, :]) == 2
            rm(file_input)
            # test output file
            @test length(readdlm(file_output, ',')[:, 1]) == num_p + 1
            @test length(readdlm(file_output, ',')[1, :]) == 3
            rm(file_output)
        else
            # test input file
            df = Arrow.Table(file_input)
            @test length(df) == 2
            @test length(df[1]) == num_p
            rm(file_input)
            # test output file
            df = Arrow.Table(file_output)
            @test length(df) == 3
            @test length(df[1]) == num_p
            rm(file_output)
        end
    end
end
