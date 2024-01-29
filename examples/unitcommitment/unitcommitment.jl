###################################################
############## Unit Commitment Model ##############
###################################################

##############
# Load Functions
##############

using MLJFlux
using Flux
using MLJ
using L2O

##############
# Parameters
##############

case_name = "case300"
date = "2017-01-01"
horizon = 2
save_file = case_name * "_" * replace(date, "-" => "_") * "_h" * string(horizon)
model_dir = joinpath(pwd(), "examples", "unitcommitment", "models")

##############
# Load Model
##############

mach = machine(joinpath(model_dir, save_file * ".jls"))

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