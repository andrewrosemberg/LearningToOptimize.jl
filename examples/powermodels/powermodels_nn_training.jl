####################################################
############ PowerModels Model Training ############
####################################################

using Arrow
using CSV
using MLJFlux
using MLUtils
using MLJBase
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
layers = [512] # [512, 256, 64] # [256, 64, 32][1024, 1024, 1024]
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

X = Float32.(Matrix(train_table[!, input_features]))
y = Float32.(Matrix(train_table[!, [:operational_cost]]))

##############
# Fit DNN Approximator
##############

# Define model and logger
lg = WandbLogger(
    project = "powermodels-obj-proxies",
    name = "$(save_file)-$(now())",
    config = Dict(
        "layers" => layers,
        "batch_size" => 32,
        "optimiser" => "ConvexRule",
        "learning_rate" => 0.01,
        "rng" => 123,
        "network_formulation" => network_formulation,
        # "lambda" => 0.00,
    ),
)

optimiser = Flux.Optimise.Adam(
    get_config(lg, "learning_rate"),
    (0.9, 0.999),
    1.0e-8,
    IdDict{Any,Any}(),
)
if icnn
    optimiser = ConvexRule(
        Flux.Optimise.Adam(
            get_config(lg, "learning_rate"),
            (0.9, 0.999),
            1.0e-8,
            IdDict{Any,Any}(),
        ),
    )
end

# optimiser= Flux.Optimise.Adam(0.01, (0.9, 0.999), 1.0e-8, IdDict{Any,Any}())
# nn = MultitargetNeuralNetworkRegressor(;
#     builder=FullyConnectedBuilder(layers),
#     rng=123,
#     epochs=5000,
#     optimiser=optimiser,
#     acceleration=CUDALibs(),
#     batch_size=32,
#     loss=relative_rmse,
# )

nn = MultitargetNeuralNetworkRegressor(;
    builder = FullyConnectedBuilder(layers),
    rng = get_config(lg, "rng"),
    epochs = 5000,
    optimiser = optimiser,
    acceleration = CUDALibs(),
    batch_size = get_config(lg, "batch_size"),
    # lambda=get_config(lg, "lambda"),
    loss = relative_rmse,
)

# Constrols

model_dir = joinpath(dirname(@__FILE__), "models")
mkpath(model_dir)

# save_control =
#     MLJIteration.skip(Save(joinpath(model_dir, save_file * ".jls")), predicate=1000)

model_path = joinpath(model_dir, save_file * ".jld2")

save_control = SaveBest(1000, model_path, 0.003)

controls = [
    Step(1),
    WithModelLossDo(save_control; stop_if_true = true),
    # NumberLimit(n=4),
    # NumberSinceBest(6),
    # PQ(; alpha=0.9, k=30),
    # GL(; alpha=4.0),
    InvalidValue(),
    # Threshold(0.003),
    TimeLimit(; t = 3),
    WithLossDo(update_loss),
    WithReportDo(update_training_loss),
    WithIterationsDo(update_epochs),
]

iterated_pipe = IteratedModel(
    model = nn,
    controls = controls,
    # resampling=Holdout(fraction_train=0.7),
    measure = relative_mae,
)

# Fit model
mach = machine(iterated_pipe, X, y)
fit!(mach; verbosity = 2)

# Finish the run
close(lg)

# # save model if final loss is better than the best loss
# if IterationControl.loss(iterated_pipe) < save_control.best_loss
#     mach = mach |> cpu

#     fitted_model = mach.fitresult.fitresult[1]

#     model_state = Flux.state(fitted_model)

#     jldsave(model_path; model_state=model_state, layers=layers, input_features=input_features)
# end
