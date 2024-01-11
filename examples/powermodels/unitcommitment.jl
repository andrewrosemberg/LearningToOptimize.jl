using LinearAlgebra
using MLJFlux
using Gurobi
using Flux
using MLJ
using L2O
using JuMP
using CUDA
using Logging
using HiGHS, Gurobi
using JuMP
using UnitCommitment
import ParametricOptInterface as POI

import UnitCommitment:
    Formulation,
    KnuOstWat2018,
    MorLatRam2013,
    ShiftFactorsFormulation

# Read benchmark instance
instance = UnitCommitment.read_benchmark(
    "matpower/case300/2017-01-01",
)
instance.time = 2
inner_solver = () -> POI.Optimizer(Gurobi.Optimizer())
upper_solver = Gurobi.Optimizer

# Construct model (using state-of-the-art defaults)
model = UnitCommitment.build_model(
    instance = instance,
    optimizer = upper_solver,
)
set_optimizer_attribute(model, "PoolSearchMode", 2)
set_optimizer_attribute(model, "PoolSolutions", 100)

bin_vars = [val for val in values(model[:is_on])]
obj_terms = objective_function(model).terms
obj_terms_gurobi = [obj_terms[var] for var in all_variables(model) if haskey(obj_terms, var)]
# gurobi_indexes = [Gurobi.column(backend(model).optimizer.model, model.moi_backend.model_to_optimizer_map[bin_vars[i].index]) for i in 1:length(bin_vars)]
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

true_ob_value = objective_value(model)
true_sol = value.(bin_vars)

# Solve model using cutting plane algorithm
include("src/cutting_planes.jl")

# upper_model, lower_bound, upper_bound, gap = cutting_planes!(model; upper_solver, inner_solver)

optimiser=Flux.Optimise.Adam()
nn = MultitargetNeuralNetworkRegressor(;
    builder=FullyConnectedBuilder([64, 32]),
    rng=123,
    epochs=100,
    optimiser=optimiser,
    acceleration=CUDALibs(),
    batch_size=32,
)

# Define the machine
X = hcat(my_storage_vars...)'[:,:]
y = convert.(Float64, my_storage_obj[:,:])

# constrols

clear() = begin
    global losses = []
    global training_losses = []
    global epochs = []
    return nothing
end

update_loss(loss) = push!(losses, loss)
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

iterated_pipe =
    IteratedModel(model=nn,
        controls=controls,
        resampling=Holdout(fraction_train=0.8),
        measure = Flux.mse
)

clear()
mach = machine(iterated_pipe, X, y)
fit!(mach; verbosity=2)

# Make predictions
predictions = convert.(Float64, predict(mach, X))
error = abs.(y .- predictions) ./ y
mean(error)

# Get upper model
upper_model, inner_2_upper_map, cons_mapping = copy_binary_model(model)

# add nn to model
bin_vars_upper = [inner_2_upper_map[var] for var in values(model[:is_on])]
obj = mach.fitresult.fitresult[1](bin_vars_upper)

aux = @variable(upper_model)
@constraint(upper_model, obj[1] <= aux)
@constraint(upper_model, aux >= 0.0)
@objective(upper_model, Min, aux)

set_optimizer(upper_model, upper_solver)
JuMP.optimize!(upper_model)

termination_status(upper_model)
value.(bin_vars_upper)
objective_value(upper_model)

mach.fitresult.fitresult[1](value.(bin_vars_upper))
abs(value(obj[1]) - true_ob_value) / true_ob_value