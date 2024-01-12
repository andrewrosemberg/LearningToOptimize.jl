###################################################
############## Unit Commitment Model ##############
###################################################

##############
# Load Packages
##############

using LinearAlgebra
using MLJFlux
using Gurobi
using Flux
using MLJ
using L2O
using JuMP
using CUDA
using Logging
using JuMP
using UnitCommitment
import ParametricOptInterface as POI
using DataFrames
using CSV
using UUIDs

import UnitCommitment:
    Formulation,
    KnuOstWat2018,
    MorLatRam2013,
    ShiftFactorsFormulation

include("src/cutting_planes.jl")

data_dir = joinpath(pwd(), "examples/unitcommitment", "data") # joinpath(dirname(@__FILE__), "data")

##############
# Solver
##############

inner_solver = () -> POI.Optimizer(Gurobi.Optimizer())
upper_solver = Gurobi.Optimizer

##############
# Load Instance
##############
case_name = "case300"
date = "2017-01-01"
horizon = 2
save_file = case_name * "_" * replace(date, "-" => "_") * "_h" * string(horizon)
instance = UnitCommitment.read_benchmark(
    joinpath("matpower", case_name, date),
)
instance.time = horizon

# Construct model (using state-of-the-art defaults)
model = UnitCommitment.build_model(
    instance = instance,
    optimizer = upper_solver,
)

# Set solver attributes
set_optimizer_attribute(model, "PoolSearchMode", 2)
set_optimizer_attribute(model, "PoolSolutions", 300)

##############
# Solve and store solutions
##############

bin_vars = vcat(
    collect(values(model[:is_on])), 
    collect(values(model[:startup])), 
    collect(values(model[:switch_on])),
    collect(values(model[:switch_off]))
)
function tuple_2_name(smb)
    str = string(smb[1])
    for i in 2:length(smb)
        str = str * "_" * string(smb[i])
    end
    return str
end
bin_vars_names = vcat(
    "is_on_" .* tuple_2_name.(collect(keys(model[:is_on]))), 
    "startup_" .* tuple_2_name.(collect(keys(model[:startup]))), 
    "switch_on_" .* tuple_2_name.(collect(keys(model[:switch_on]))),
    "switch_off_" .* tuple_2_name.(collect(keys(model[:switch_off])))
)

@assert all([is_binary(var) for var in bin_vars])
@assert length(bin_vars) == length(all_binary_variables(model))

obj_terms = objective_function(model).terms
obj_terms_gurobi = [obj_terms[var] for var in all_variables(model) if haskey(obj_terms, var)]
num_bin_var = length(bin_vars)
num_all_var = num_variables(model)

global my_storage_vars = []
global my_storage_obj = []
global is_relaxed = []
global non_optimals = []
function my_callback_function(cb_data, cb_where::Cint)
    # You can select where the callback is run
    if cb_where == GRB_CB_MIPNODE
        resultobj = Ref{Cint}()
        GRBcbget(cb_data, cb_where, GRB_CB_MIPNODE_STATUS, resultobj)
        if resultobj[] != GRB_OPTIMAL
            push!(non_optimals, resultobj[])
            return  # Solution is something other than optimal.
        end
        gurobi_indexes_all = [Gurobi.column(backend(model).optimizer.model, model.moi_backend.model_to_optimizer_map[var.index]) for var in all_variables(model) if haskey(obj_terms, var)]
        gurobi_indexes_bin = [Gurobi.column(backend(model).optimizer.model, model.moi_backend.model_to_optimizer_map[bin_vars[i].index]) for i in 1:length(bin_vars)]
        resultP = Vector{Cdouble}(undef, num_all_var)
        GRBcbget(cb_data, cb_where, GRB_CB_MIPNODE_REL, resultP)
        push!(my_storage_vars, resultP[gurobi_indexes_bin])
        # Get the objective value
        push!(my_storage_obj, dot(obj_terms_gurobi, resultP[gurobi_indexes_all]))
        # mark as relaxed
        push!(is_relaxed, 1)
        return
    end
    if cb_where == GRB_CB_MIPSOL
        # Before querying `callback_value`, you must call:
        Gurobi.load_callback_variable_primal(cb_data, cb_where)
        # Get the values of the variables
        x = [callback_value(cb_data, var) for var in bin_vars]
        # push
        push!(my_storage_vars, x)
        # Get the objective value
        obj = Ref{Cdouble}()
        GRBcbget(cb_data, cb_where, GRB_CB_MIPSOL_OBJ, obj)
        # push
        push!(my_storage_obj, obj[])
        # mark as not relaxed
        push!(is_relaxed, 0)
        return
    end
    return
end
MOI.set(model, Gurobi.CallbackFunction(), my_callback_function)

# JuMP.optimize!(model)
UnitCommitment.optimize!(model)
is_relaxed = findall(x -> x == 1, is_relaxed)

# Data
X = hcat(my_storage_vars...)'[:,:]
y = convert.(Float64, my_storage_obj[:,:])

batch_id = uuid1()

# Save solutions
instances_ids = [uuid1() for i in 1:length(my_storage_vars)]
df_in = DataFrame(Dict(Symbol.(bin_vars_names) .=> eachcol(X)))
df_in.id = instances_ids
df_out = DataFrame(Dict(:objective => y[:,1]))
df_out.id = instances_ids
# time,status,primal_status,dual_status
df_out.time = fill(0.0, length(instances_ids))
df_out.status = fill("OPTIMAL", length(instances_ids))
df_out.primal_status = fill("FEASIBLE_POINT", length(instances_ids))
df_out.dual_status = fill("FEASIBLE_POINT", length(instances_ids))

CSV.write(joinpath(data_dir, save_file * "_input_" * string(batch_id) * ".csv"), df_in)
CSV.write(joinpath(data_dir, save_file * "_output_" * string(batch_id) * ".csv"), df_out)

true_ob_value = objective_value(model)
true_sol = value.(bin_vars)

# Solve model using cutting plane algorithm
# upper_model, lower_bound, upper_bound, gap = cutting_planes!(model; upper_solver, inner_solver)

##############
# Enhance dataset
##############
# Get upper model
inner_model = model
MOI.set(inner_model, Gurobi.CallbackFunction(), nothing)
upper_model, inner_2_upper_map, cons_mapping = copy_binary_model(inner_model)

# delete binary constraints from inner model
delete_binary_terms!(inner_model; delete_objective=false)
# add deficit constraints
add_deficit_constraints!(inner_model)
# link binary variables from upper to inner model
upper_2_inner = fix_binary_variables!(inner_model, inner_2_upper_map)
# get parameters from inner model in the right order
u_inner = [upper_2_inner[inner_2_upper_map[var]] for var in bin_vars]
# set names
set_name.(u_inner, bin_vars_names)
# set solver
set_optimizer(inner_model, inner_solver)
# Parameter values
num_s = 1000
parameter_values = Dict(u_inner .=> [rand(num_s) for i in 1:length(u_inner)])

# The iterator
problem_iterator = ProblemIterator(parameter_values)
input_file = "input_" * save_file
save(problem_iterator, joinpath(data_dir, input_file), CSVFile)
input_file = input_file * "." * string(CSVFile)
# CSV recorder to save the optimal primal and dual decision values
output_file = "output_" * save_file
recorder = Recorder{CSVFile}(joinpath(data_dir, output_file); model=inner_model)
output_file = output_file * "." * string(CSVFile)

# Finally solve all problems described by the iterator
solve_batch(problem_iterator, recorder)

# Read input and output data
input_data = CSV.read(joinpath(data_dir, input_file), DataFrame)
output_data = CSV.read(joinpath(data_dir, output_file), DataFrame)

# Separate input and output variables
output_variables = output_data[!, :objective]
input_features = innerjoin(input_data, output_data[!, [:id]], on = :id)[!, Symbol.(bin_vars_names)] # just use success solves


##############
# Fit DNN approximator
##############

# optimiser=Flux.Optimise.Adam()
optimiser=ConvexRule(
    Flux.Optimise.Adam(0.01, (0.9, 0.999), 1.0e-8, IdDict{Any,Any}())
)
nn = MultitargetNeuralNetworkRegressor(;
    builder=FullyConnectedBuilder([1024, 1024, 300, 64, 32]), # [1024, 300, 64, 32]
    rng=123,
    epochs=100,
    optimiser=optimiser,
    acceleration=CUDALibs(),
    batch_size=24,
)

X = vcat(X, Matrix(input_features))
y = vcat(y, output_variables)

# Constrols

clear() = begin
    global losses = []
    global training_losses = []
    global epochs = []
    return nothing
end

function update_loss(loss)
    @info "Loss: $loss"
    push!(losses, loss)
end
update_training_loss(report) =
    push!(training_losses,
          report.training_losses[end])
update_epochs(epoch) = push!(epochs, epoch)

controls=[Step(1),
    NumberSinceBest(6),
    InvalidValue(),
    TimeLimit(5/60),
    WithLossDo(update_loss),
    WithReportDo(update_training_loss),
    WithIterationsDo(update_epochs)
]

# WIP
function SobolevLoss_mse(x, y)
    o_term = Flux.mse(x, y[:, 1])
    d_term = Flux.mse(gradient( ( _x ) -> sum(layer( _x )), x), y[:, 2:end])
    return o_term + d_term * 0.1
end

iterated_pipe =
    IteratedModel(model=nn,
        controls=controls,
        resampling=Holdout(fraction_train=0.8),
        measure = l2,
)

# Fit model
clear()
mach = machine(iterated_pipe, X, y)
fit!(mach; verbosity=2)

# Make predictions
predictions = convert.(Float64, predict(mach, X))
error = abs.(y .- predictions) ./ y
mean(error)

# Derivatives
layer = mach.fitresult.fitresult[1]
gradient( ( _x ) -> sum(layer( _x )), X')


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