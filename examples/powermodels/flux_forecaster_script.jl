using Arrow
using CSV
using MLJFlux
using MLUtils
using Flux
using MLJ
using DataFrames
using PowerModels
using L2O
using Random

# Paths
case_name = "pglib_opf_case300_ieee" # pglib_opf_case300_ieee # pglib_opf_case5_pjm
network_formulation = DCPPowerModel # SOCWRConicPowerModel # DCPPowerModel
filetype = ArrowFile # ArrowFile # CSVFile
path_dataset = joinpath(pwd(), "examples", "powermodels", "data")
case_file_path = joinpath(path_dataset, case_name)
case_file_path_output = joinpath(case_file_path, "output", string(network_formulation))
case_file_path_input = joinpath(case_file_path, "input", "train")

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
# batch_ids = [split(split(file, "_")[end], ".")[1] for file in file_ins]

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

# Split the data into training and test sets
# seed random number generator

# Separate input and output variables & ignore id time status primal_status dual_status
train_table = innerjoin(input_data, output_data[!, [:id, :operational_cost]]; on=:id)

input_features = names(joined_table[!, Not([:id, :operational_cost])])

X = Float32.(Matrix(train_table[!, input_features]))
y = Float32.(train_table[!, :operational_cost])

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
