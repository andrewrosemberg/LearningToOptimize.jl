####################################################
############ PowerModels Model Testing #############
####################################################

using Arrow
using CSV
using Flux
using DataFrames
using PowerModels
using L2O
using JLD2
using Statistics

##############
# Parameters
##############
case_name = ARGS[1] # case_name="pglib_opf_case300_ieee" # pglib_opf_case5_pjm
network_formulation = ARGS[2] # network_formulation=DCPPowerModel SOCWRConicPowerModel DCPPowerModel
filetype = ArrowFile # ArrowFile # CSVFile
path_dataset = joinpath(dirname(@__FILE__), "data")
case_file_path = joinpath(path_dataset, case_name)
case_file_path_output = joinpath(case_file_path, "output", string(network_formulation))
case_file_path_input = joinpath(case_file_path, "input", "test")
save_file = "$(case_name)_$(network_formulation)"


##############
# Load Data
##############
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
output_data = output_data[output_data.operational_cost .> 10, :]

# match
test_table = innerjoin(input_data, output_data[!, [:id, :operational_cost]]; on=:id)

##############
# Load DNN Approximator
##############

model_dir = joinpath(dirname(@__FILE__), "models")

models_list = readdir(model_dir)
models_list = filter(x -> occursin(".jld2", x), models_list)
models_list = filter(x -> occursin(string(network_formulation), x), models_list)

flux_models = Array{Chain}(undef, length(models_list))
layers = Array{Vector}(undef, length(models_list))
type = Array{String}(undef, length(models_list))
global input_features = 0
for (i, model) in enumerate(models_list)
    type[i] = occursin("icnn", model) ? "ICNN" : "DNN"
    model_save = JLD2.load(joinpath(model_dir, model))
    model_state = model_save["model_state"]
    global input_features = model_save["input_features"]
    layers[i] = model_save["layers"]
    input_size = length(input_features)
    flux_model = Chain(FullyConnected(input_size, layers[i], 1))
    Flux.loadmodel!(flux_model, model_state)
    flux_models[i] = flux_model
end

##############
# Prepare Data
##############

X = Float32.(Matrix(test_table[!, input_features]))
y = Float32.(Matrix(test_table[!, [:operational_cost]]))

##############
# Predict
##############
mae_convex_hull = Array{Float32}(undef, length(flux_models))
mae_out_convex_hull = Array{Float32}(undef, length(flux_models))
worst_case_convex_hull = Array{Float32}(undef, length(flux_models))
worst_case_out_convex_hull = Array{Float32}(undef, length(flux_models))
std_convex_hull = Array{Float32}(undef, length(flux_models))
std_out_convex_hull = Array{Float32}(undef, length(flux_models))

for (i, flux_model) in enumerate(flux_models)
    error_vec = (y .- flux_model(X')') ./ y
    error_vec_in_chull = error_vec[test_table.in_train_convex_hull .== 1]
    error_vec_out_chull = error_vec[test_table.in_train_convex_hull .== 0]
    mae_convex_hull[i] = mean(abs.(error_vec_in_chull))
    mae_out_convex_hull[i] = mean(abs.(error_vec_out_chull))
    worst_case_convex_hull[i] = maximum(abs.(error_vec_in_chull))
    worst_case_out_convex_hull[i] = maximum(abs.(error_vec_out_chull))
    std_convex_hull[i] = std(error_vec_in_chull)
    std_out_convex_hull[i] = std(error_vec_out_chull)
end

# table
results = DataFrame(
    type = type,
    layers = layers,
    mae_convex_hull = mae_convex_hull,
    mae_out_convex_hull = mae_out_convex_hull,
    worst_case_convex_hull = worst_case_convex_hull,
    worst_case_out_convex_hull = worst_case_out_convex_hull,
    std_convex_hull = std_convex_hull,
    std_out_convex_hull = std_out_convex_hull,
)

# save
results_dir = joinpath(dirname(@__FILE__), "results")
mkpath(results_dir)
save_file = joinpath(results_dir, "$(save_file)_results.csv")

CSV.write(save_file, results)