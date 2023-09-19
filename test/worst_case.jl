
"""
    test_worst_case_problem_iterator(path::AbstractString)

Test dataset generation using the worst case problem iterator for different filetypes.
"""
function test_worst_case_problem_iterator(path::AbstractString)
    @testset "Worst Case Generation Type: $filetype" for filetype in [CSVFile, ArrowFile]
        # The problem to iterate over
        function optimizer_factory()
            return () -> Ipopt.Optimizer()
        end
        parameter_factory = (model) -> [@variable(model, _p)]
        function primal_builder!(model, parameters; recorder=nothing)
            @variable(model, x)
            @constraint(model, cons, x + parameters[1] >= 3)
            @objective(model, Min, 2x)

            if !isnothing(recorder)
                set_primal_variable!(recorder, [x])
                set_dual_variable!(recorder, [cons])
            end
        end
        function set_iterator!(model, parameters, idx)
            @constraint(model, parameters[1] <= 1.0 * idx)
            @constraint(model, parameters[1] >= 0.0)
        end
        num_p = 10
        problem_iterator = WorstCaseProblemIterator(
            [uuid1() for _ in 1:num_p],
            parameter_factory,
            primal_builder!,
            set_iterator!,
            optimizer_factory,
        )

        # file_names
        file_input = joinpath(path, "test_input.$(string(filetype))")
        file_output = joinpath(path, "test_output.$(string(filetype))")

        # The recorder
        recorder = Recorder{filetype}(
            file_output; filename_input=file_input,
            primal_variables=[], dual_variables=[]
        )
        
        # Solve all problems and record solutions
        solve_batch(problem_iterator, recorder)

        # Check if file exists and has the correct number of rows and columns
        @test isfile(file_input)
        @test isfile(file_output)
        if filetype == CSVFile
            # test input file
            @test length(readdlm(file_input, ',')[:, 1]) == num_p + 1
            @test length(readdlm(file_input, ',')[1, :]) == 2
            rm(file_input)
            # test output file
            @test length(readdlm(file_output, ',')[:, 1]) == num_p + 1
            @test length(readdlm(file_output, ',')[1, :]) == 4
            rm(file_output)
        else
            # test input file
            df = Arrow.Table(file_input)
            @test length(df) == 2
            @test length(df[1]) == num_p
            rm(file_input)
            # test output file
            df = Arrow.Table(file_output)
            @test length(df) == 4
            @test length(df[1]) == num_p
            rm(file_output)
        end
    end
end
