using TestEnv
TestEnv.activate()

using Arrow
using CSV
using MLJFlux
using MLUtils
using Flux
using MLJ
using DataFrames
using PowerModels
using L2O

# Paths
case_name = "pglib_opf_case300_ieee" # pglib_opf_case300_ieee # pglib_opf_case5_pjm
network_formulation = DCPPowerModel # SOCWRConicPowerModel # DCPPowerModel
filetype = ArrowFile # ArrowFile # CSVFile
path_dataset = joinpath(pwd(), "examples", "powermodels", "data")
case_file_path = joinpath(path_dataset, case_name)
case_file_path_output = joinpath(case_file_path, "output", string(network_formulation))
case_file_path_input = joinpath(case_file_path, "input")

# Load input and output data tables
iter_files_in = readdir(joinpath(case_file_path_input))
iter_files_in = filter(x -> occursin(string(filetype), x), iter_files_in)
file_ins = [
    joinpath(case_file_path_input, file) for file in iter_files_in if occursin("input", file)
]
iter_files_out = readdir(joinpath(case_file_path_output))
iter_files_out = filter(x -> occursin(string(filetype), x), iter_files_out)
file_outs = [
    joinpath(case_file_path_output, file) for file in iter_files_out if occursin("output", file)
]
batch_ids = [split(split(file, "_")[end], ".")[1] for file in file_ins]

# Load input and output data tables
if filetype === ArrowFile
    input_table_train = Arrow.Table(file_ins)
    output_table_train = Arrow.Table(file_outs)
else
    input_table_train = CSV.read(file_ins[train_idx], DataFrame)
    output_table_train = CSV.read(file_outs[train_idx], DataFrame)
end

# Convert to dataframes
input_data = DataFrame(input_table_train)
output_data = DataFrame(output_table_train)

train_idx, test_idx = splitobs(1:size(input_data, 1), at=(0.7), shuffle=true)

input_data_train = input_data[train_idx, :]
output_data_train = output_data[train_idx, :]

input_data_test = input_data[test_idx, :]
output_data_test = output_data[test_idx, :]

using Gurobi
inhull = inconvexhull(Matrix(input_data_train[!, Not(:id)]), Matrix(input_data_test[1:10, Not(:id)]), Gurobi.Optimizer)

# Separate input and output variables
output_variables_train = output_data_train[!, Not(:id)]
input_features_train = innerjoin(input_data_train, output_data_train[!, [:id]]; on=:id)[
    !, Not(:id)
] # just use success solves

num_loads = floor(Int, size(input_features_train, 2) / 2)
total_volume = [
    sum(
        sqrt(input_features_train[i, l]^2 + input_features_train[i, l + num_loads]^2) for
        l in 1:num_loads
    ) for i in 1:size(input_features_train, 1)
]

output_variables_test = output_data_test[!, Not(:id)]
input_features_test = innerjoin(input_data_test, output_data_test[!, [:id]]; on=:id)[
    !, Not(:id)
] # just use success solves

# Define model
model = MultitargetNeuralNetworkRegressor(;
    builder=FullyConnectedBuilder([64, 32]),
    rng=123,
    epochs=20,
    optimiser=ConvexRule(
        Flux.Optimise.Adam(0.001, (0.9, 0.999), 1.0e-8, IdDict{Any,Any}())
    ),
)

# Define the machine
mach = machine(model, input_features_train, output_variables_train)
fit!(mach; verbosity=2)

# Make predictions
predictions = predict(mach, input_features)

# Calculate the error
error = Flux.mse(predictions, output_variables_test)
