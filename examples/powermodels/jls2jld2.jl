using Arrow
using CSV
using MLJFlux
using MLUtils
using Flux
using MLJ
using CUDA
using DataFrames
using PowerModels
using LearningToOptimize
using Random
using JLD2
using Wandb, Dates, Logging

include(joinpath(dirname(dirname(@__FILE__)), "training_utils.jl")) # include("../training_utils.jl")

##############
# Parameters
##############
case_name = ARGS[1] # case_name="pglib_opf_case300_ieee" # pglib_opf_case5_pjm
network_formulation = ARGS[2] # network_formulation=ACPPowerModel SOCWRConicPowerModel DCPPowerModel
icnn = parse(Bool, ARGS[3]) # icnn=true # false
filetype = ArrowFile # ArrowFile # CSVFile
layers = [512, 256, 64] # [256, 64, 32]
path_dataset = joinpath(dirname(@__FILE__), "data")
case_file_path = joinpath(path_dataset, case_name)
case_file_path_output = joinpath(case_file_path, "output", string(network_formulation))
case_file_path_input = joinpath(case_file_path, "input", "train")
save_file = if icnn
    "$(case_name)_$(network_formulation)_$(replace(string(layers), ", " => "_"))_icnn"
else
    "$(case_name)_$(network_formulation)_$(replace(string(layers), ", " => "_"))_dnn"
end

##############
# Load Data
##############

iter_files_in = readdir(joinpath(case_file_path_input))
iter_files_in = filter(x -> occursin(string(filetype), x), iter_files_in)
file_ins = [
    joinpath(case_file_path_input, file) for
    file in iter_files_in if occursin("input", file)
]
iter_files_out = readdir(joinpath(case_file_path_output))
iter_files_out = filter(x -> occursin(string(filetype), x), iter_files_out)
file_outs = [
    joinpath(case_file_path_output, file) for
    file in iter_files_out if occursin("output", file)
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

# filter out rows with 0.0 operational_cost (i.e. inidicative of numerical issues)
output_data = output_data[output_data.operational_cost.>10, :]

# match
train_table = innerjoin(input_data, output_data[!, [:id, :operational_cost]]; on = :id)

input_features = names(train_table[!, Not([:id, :operational_cost])])

##############
# Load JLS Model
##############
num = 3
model_dir = joinpath(dirname(@__FILE__), "models")
mach = machine(joinpath(model_dir, save_file * "$num.jls"))

##############
# Save JLD2 Model
##############
model = mach.fitresult[1]
model_state = Flux.state(model)
jldsave(
    joinpath(model_dir, save_file * ".jld2"),
    model_state = model_state,
    input_features = input_features,
    layers = mach.model.builder.hidden_sizes,
)
