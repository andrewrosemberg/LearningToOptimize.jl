###################################################
############## Unit Commitment Model ##############
###################################################

##############
# Load Functions
##############

# using MLJFlux
using Flux
# using MLJ
using L2O
# using CUDA
# using Wandb
using DataFrames
using JLD2
using Arrow

# include(joinpath(pwd(), "examples", "unitcommitment", "training_utils.jl"))
include(joinpath(pwd(), "examples", "unitcommitment", "bnb_dataset.jl"))

##############
# Parameters
##############

case_name = "case300"
date = "2017-01-01"
horizon = 2
save_file = case_name * "_" * replace(date, "-" => "_") * "_h" * string(horizon)
model_dir = joinpath(pwd(), "examples", "unitcommitment", "models")
data_file_path = joinpath(pwd(), "examples", "unitcommitment", "data")
case_file_path = joinpath(data_file_path, case_name, date, "h"*string(horizon))

##############
# Load Model
##############
# mach = machine(joinpath(model_dir, save_file * ".jlso"))

model_state = JLD2.load(joinpath(model_dir, save_file * ".jld2"), "model_state")

##############
# Load Data
##############

file_in = readdir(joinpath(case_file_path, "input"), join=true)[1]
batch_id = split(split(split(file_in, "/")[end], "_input_")[2], ".")[1]
file_outs = readdir(joinpath(case_file_path, "output"), join=true)
file_out = [file for file in file_outs if occursin(batch_id, file)][1]

input_table = Arrow.Table(file_in) |> DataFrame
output_table = Arrow.Table(file_out) |> DataFrame

num_input = size(input_table, 2) - 1

train_table = innerjoin(input_table, output_table[!, [:id, :objective]]; on=:id)
input_features = train_table[!, Not([:id, :objective])]
X = Float32.(Matrix(input_features))
y = Float32.(Matrix(train_table[!, [:objective]]))

##############
# Load Instance
##############
instance = UnitCommitment.read_benchmark(
    joinpath("matpower", case_name, date),
)
instance.time = horizon

# impose load
for i in 1:length(instance.buses)
    bus = instance.buses[i]
    bus_name = bus.name
    for h in 1:horizon
        bus.load[h] = input_table[1, Symbol("load_" * bus_name * "_" * string(h))]
    end
end

# build model
model = build_model_uc(instance)
bin_vars, bin_vars_names = bin_variables_retriever(model)

##############
# Build DNN
##############

input_size = size(input_table, 2) - 1
flux_model = FullyConnected(input_size, [1024, 512, 64], 1)
Float64(relative_mae(flux_model(X'), y'))


# Remove binary constraints
upper_model, inner_2_upper_map, cons_mapping = copy_binary_model(model)

##############
# Solve using DNN approximator
##############

function NNlib.relu(ex::AffExpr)
    model = owner_model(ex)
    relu_out = @variable(model, lower_bound = 0.0)
    @constraint(model, relu_out >= ex)
    return relu_out
end

# add nn to model
bin_vars_upper = [inner_2_upper_map[var] for var in bin_vars]
obj = mach.fitresult.fitresult[1](bin_vars_upper)

aux = @variable(upper_model)
@constraint(upper_model, obj[1] <= aux)
@constraint(upper_model, aux >= 0.0)
@objective(upper_model, Min, aux)

set_optimizer(upper_model, upper_solver)
JuMP.optimize!(upper_model)

termination_status(upper_model)
sol_bin_vars = round.(Int, value.(bin_vars_upper))
objective_value(upper_model)

##############
# Compare
##############
# Compare using approximator
mach.fitresult.fitresult[1](sol_bin_vars .* 1.0)
abs(value(obj[1]) - true_ob_value) / true_ob_value

# Compare using foward pass
for i in 1:length(bin_vars_upper)
    fix(bin_vars[i], sol_bin_vars[i])
end

JuMP.optimize!(model)
abs(objective_value(model) - true_ob_value) / true_ob_value