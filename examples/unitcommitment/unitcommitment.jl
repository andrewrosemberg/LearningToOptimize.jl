###################################################
############## Unit Commitment Model ##############
###################################################

##############
# Load Functions
##############

# using MLJFlux
using Flux
# using MLJ
# using LearningToOptimize
# using CUDA
# using Wandb
using DataFrames
using JLD2
using Arrow
using Gurobi

include(joinpath(pwd(), "src/cutting_planes.jl"))
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

upper_solver = Gurobi.Optimizer

##############
# Load Model
##############
# mach = machine(joinpath(model_dir, save_file * ".jlso"))

model_save = JLD2.load(joinpath(model_dir, save_file * ".jld2"))

model_state = model_save["model_state"]

input_features = model_save["input_features"]

layers = model_save["layers"]

##############
# Load Data
##############

file_in = readdir(joinpath(case_file_path, "input"), join=true)[1]
batch_id = split(split(split(file_in, "/")[end], "_input_")[2], ".")[1]
file_outs = readdir(joinpath(case_file_path, "output"), join=true)
file_out = [file for file in file_outs if occursin(batch_id, file)][1]

input_table = Arrow.Table(file_in) |> DataFrame
output_table = Arrow.Table(file_out) |> DataFrame

train_table = innerjoin(input_table, output_table[!, [:id, :objective]]; on=:id)
X = Float32.(Matrix(train_table[!, Symbol.(input_features)]))
y = Float32.(Matrix(train_table[!, [:objective]]))

true_ob_value = output_table[output_table.status .== "OPTIMAL", :objective][1]

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

# Remove binary constraints
upper_model, inner_2_upper_map, cons_mapping = copy_binary_model(model)

##############
# Build DNN
##############

input_size = length(input_features)
flux_model = Chain(FullyConnected(input_size, layers, 1))
Flux.loadmodel!(flux_model, model_state)
Float64(relative_mae(flux_model(X'), y'))

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
bin_vars_upper = Array{Any}(undef, length(input_features))
for i in 1:length(input_features)
    feature = input_features[i]
    if occursin("load", feature)
        bin_vars_upper[i] = input_table[1, Symbol(feature)]
    else
        idx = findfirst(isequal(feature), bin_vars_names)
        bin_vars_upper[i] = inner_2_upper_map[bin_vars[idx]]
    end
end

obj = flux_model(bin_vars_upper)

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
# mach.fitresult.fitresult[1](sol_bin_vars .* 1.0)
abs(value(obj[1]) - true_ob_value) / true_ob_value

# Compare using foward pass
for var in bin_vars
    fix(var, value(inner_2_upper_map[var]))
end

JuMP.optimize!(model)
abs(objective_value(model) - true_ob_value) / true_ob_value