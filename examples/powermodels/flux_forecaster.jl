using Flux
using CSV
using DataFrames
using L2O

function test_flux_forecaster(file_in::AbstractString, file_out::AbstractString)
    @testset "Flux.jl" begin
        # read input and output data
        input_data = CSV.read(file_in, DataFrame)
        output_data = CSV.read(file_out, DataFrame)

        # Separate input and output variables
        output_variables = output_data[:, 2:end]
        input_features = input_data[output_data[:, 1], 2:end] # just use success solves

        # Define model
        model = Chain(
            Dense(size(input_features, 2), 64, relu),
            Dense(64, 32, relu),
            Dense(32, size(output_variables, 2))
        )

        # Define loss function
        loss(x, y) = Flux.mse(model(x), y)

        # Convert the data to matrices
        input_features = Matrix(input_features)'
        output_variables = Matrix(output_variables)'

        # Define the optimizer
        optimizer = Flux.ADAM()

        # Train the model
        Flux.train!(loss, Flux.params(model), [(input_features, output_variables)], optimizer)

        # Make predictions
        predictions = model(input_features)
        @test predictions isa Matrix
    end
end