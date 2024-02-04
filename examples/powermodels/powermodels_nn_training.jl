####################################################
############ PowerModels Model Training ############
####################################################

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
using JLD2

##############
# Parameters
##############
case_name = "pglib_opf_case300_ieee" # pglib_opf_case300_ieee # pglib_opf_case5_pjm
network_formulation = DCPPowerModel # SOCWRConicPowerModel # DCPPowerModel
filetype = ArrowFile # ArrowFile # CSVFile
path_dataset = joinpath(pwd(), "examples", "powermodels", "data")
case_file_path = joinpath(path_dataset, case_name)
case_file_path_output = joinpath(case_file_path, "output", string(network_formulation))
case_file_path_input = joinpath(case_file_path, "input", "train")

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

# match
train_table = innerjoin(input_data, output_data[!, [:id, :operational_cost]]; on=:id)

input_features = names(joined_table[!, Not([:id, :operational_cost])])

X = Float32.(Matrix(train_table[!, input_features]))
y = Float32.(train_table[!, :operational_cost])

##############
# Fit DNN Approximator
##############

# Define model and logger
layers = [1024, 512, 64] # [1024, 300, 64, 32] , [1024, 1024, 300, 64, 32]
lg = WandbLogger(
    project = "unit_commitment_proxies",
    name = "$(case_name)-$(now())",
    config = Dict(
        "layers" => layers,
        "batch_size" => 32,
        "optimiser" => "ConvexRule",
        "learning_rate" => 0.01,
        "rng" => 123,
        # "lambda" => 0.00,
    )
)

optimiser=ConvexRule(
    Flux.Optimise.Adam(get_config(lg, "learning_rate"), (0.9, 0.999), 1.0e-8, IdDict{Any,Any}())
)

nn = MultitargetNeuralNetworkRegressor(;
    builder=FullyConnectedBuilder(layers),
    rng=get_config(lg, "rng"),
    epochs=5000,
    optimiser=optimiser,
    acceleration=CUDALibs(),
    batch_size=get_config(lg, "batch_size"),
    # lambda=get_config(lg, "lambda"),
    loss=relative_rmse,
)


# Constrols

model_dir = joinpath(dirname(@__FILE__), "models")
mkpath(model_dir)

save_control =
    MLJIteration.skip(Save(joinpath(model_dir, save_file * ".jls")), predicate=3)

controls=[Step(2),
    NumberSinceBest(6),
    # PQ(; alpha=0.9, k=30),
    GL(; alpha=4.0),
    InvalidValue(),
    TimeLimit(; t=1),
    save_control,
    WithLossDo(update_loss),
    WithReportDo(update_training_loss),
    WithIterationsDo(update_epochs)
]

iterated_pipe =
    IteratedModel(model=nn,
        controls=controls,
        resampling=Holdout(fraction_train=0.7),
        measure = relative_mae,
)

# Fit model
mach = machine(iterated_pipe, X, y)
fit!(mach; verbosity=2)

# Finish the run
close(lg)

# save model
mach = mach |> cpu

fitted_model = mach.fitresult.fitresult[1]

model_state = Flux.state(fitted_model)

jldsave(joinpath(model_dir, save_file * ".jld2"); model_state=model_state, layers=layers, input_features=input_features)
