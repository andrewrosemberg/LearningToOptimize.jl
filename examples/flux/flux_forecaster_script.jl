using TestEnv
TestEnv.activate()

using Arrow
using Flux
using DataFrames
using PowerModels
using L2O

# Paths
case_name = "pglib_opf_case300_ieee" # pglib_opf_case300_ieee # pglib_opf_case5_pjm
network_formulation = SOCWRConicPowerModel # SOCWRConicPowerModel # DCPPowerModel
filetype = ArrowFile # ArrowFile # CSVFile
path_dataset = joinpath(pwd(), "examples", "powermodels", "data")
case_file_path = joinpath(path_dataset, case_name, string(network_formulation))

# Load input and output data tables
iter_files = readdir(joinpath(case_file_path))
iter_files = filter(x -> occursin(string(ArrowFile), x), iter_files)
file_ins = [joinpath(case_file_path, file) for file in iter_files if occursin("input", file)]
file_outs = [joinpath(case_file_path, file) for file in iter_files if occursin("output", file)]
batch_ids = [split(split(file, "_")[end], ".")[1] for file in file_ins]

# Load input and output data tables
train_idx = [1]
test_idx = [2]

input_table_train = Arrow.Table(file_ins[train_idx])
output_table_train = Arrow.Table(file_outs[train_idx])

input_table_test = Arrow.Table(file_ins[test_idx])
output_table_test = Arrow.Table(file_outs[test_idx])

# Convert to dataframes
input_data_train = DataFrame(input_table_train)
output_data_train = DataFrame(output_table_train)

input_data_test = DataFrame(input_table_test)
output_data_test = DataFrame(output_table_test)

# Separate input and output variables
output_variables_train = output_data_train[!, Not(:id)]
input_features_train = innerjoin(input_data_train, output_data_train[!, [:id]], on = :id)[!, Not(:id)] # just use success solves

num_loads = floor(Int,size(input_features_train,2)/2)
total_volume=[sum(sqrt(input_features_train[i,l]^2 + input_features_train[i,l+num_loads]^2) for l in 1:num_loads) for i in 1:size(input_features_train,1) ] 

output_variables_test = output_data_test[!, Not(:id)]
input_features_test = innerjoin(input_data_test, output_data_test[!, [:id]], on = :id)[!, Not(:id)] # just use success solves

# Define model
model = MultitargetNeuralNetworkRegressor(
    builder=FullyConnectedBuilder([64,32]),
    rng=123,
    epochs=20,
    optimiser=ConvexRule(Flux.Optimise.Adam(0.001, (0.9, 0.999), 1.0e-8, IdDict{Any, Any}()))
)

# Define the machine
mach = machine(model, input_features_train, output_variables_train)
fit!(mach, verbosity=2)

# Make predictions
predictions = predict(mach, input_features)

# Calculate the error
error = Flux.mse(predictions, output_variables_test)
