using Flux
using CSV
using DataFrames

path = joinpath(pwd(), "examples", "powermodels")

# Dataset generation
include(joinpath(path, "pglib_datagen.jl"))

# Define test case from pglib
case_name = "pglib_opf_case5_pjm.m"

# Define number of problems
num_p = 10

# Generate dataset
success_solves, number_variables = generate_dataset_pglib(
    path, case_name; num_p=num_p
)

# read input and output data
input_data = CSV.read(case_name * "_input.csv", DataFrame)
output_data = CSV.read(case_name * "_output.csv", DataFrame)

# Separate input and output variables
input_features = input_data[2:end, 2:end]
output_variables = output_data[2:end, 2:end]

# Define model
model = Chain(
    Dense(size(input_features, 2), 64, relu),
    Dense(64, 32, relu),
    Dense(32, size(output_variables, 2))
)

# Define loss function
loss(x, y) = Flux.mse(model(x), y)

# Convert the data to matrices
input_features = Matrix(input_features)
output_variables = Matrix(output_variables)

# Define the optimizer
optimizer = Flux.ADAM()

# Train the model
Flux.train!(loss, Flux.params(model), [(input_features, output_variables)], optimizer)

# Make predictions
predictions = model(input_features)


