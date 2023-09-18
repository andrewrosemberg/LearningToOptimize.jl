using PowerModels
using Ipopt
using PGLib
using JuMP
using Dualization
import JuMP.MOI as MOI

function _moi_set_upper_bound(
    moi_backend,
    idx::MOI.VariableIndex,
    upper::Number,
)
    new_set = MOI.LessThan(upper)

    JuMP._moi_add_constraint(moi_backend, idx, new_set)

    return
end

function _moi_set_lower_bound(
    moi_backend,
    idx::MOI.VariableIndex,
    lower::Number,
)
    new_set = MOI.GreaterThan(lower)

    JuMP._moi_add_constraint(moi_backend, idx, new_set)

    return
end

# Load data
case_name = "pglib_opf_case5_pjm"
matpower_case_name = case_name * ".m"
network_data = pglib(matpower_case_name)

# Case Parameters
network_formulation=DCPPowerModel
original_load = [l["pd"] for l in values(network_data["load"])]
max_total_load = sum(original_load) * 1.1

# Create model
solver = Ipopt.Optimizer
model = Model(solver)
load_var = @variable(model, load_var[i=1:length(original_load)])
for (i, l) in enumerate(values(network_data["load"]))
    l["pd"] = load_var[i]
end

# Instantiate the model
pm = instantiate_model(
    network_data,
    network_formulation,
    PowerModels.build_opf;
    setting=Dict("output" => Dict("duals" => true)),
    jump_model=model,
)

# Solve the model
JuMP.optimize!(model)

optimal_cost = JuMP.objective_value(model)

# Dualize the model
dual_st = Dualization.dualize(JuMP.backend(model), 
    variable_parameters = MOI.VariableIndex[i.index for i in load_var]
)

dual_model = dual_st.dual_model
primal_dual_map = dual_st.primal_dual_map

for (i,l) in enumerate(load_var)
    _moi_set_upper_bound(dual_model, primal_dual_map.primal_parameter[l.index], max_total_load)
    _moi_set_lower_bound(dual_model, primal_dual_map.primal_parameter[l.index], 0.0)
end

jump_dual_model = JuMP.Model()
map_moi_to_jump = MOI.copy_to(JuMP.backend(jump_dual_model), dual_model)
set_optimizer(jump_dual_model, solver)

load_dual_indexes = [map_moi_to_jump[primal_dual_map.primal_parameter[l.index]] for l in load_var]

JuMP.optimize!(jump_dual_model)

optimal_dual_cost = JuMP.objective_value(jump_dual_model)

optimal_loads = [MOI.get(JuMP.backend(jump_dual_model), MOI.VariablePrimal(), l) for l in load_dual_indexes]

cost_diff = (optimal_dual_cost - optimal_cost) / optimal_cost

load_diff = sum(abs.(optimal_loads - original_load)) ./ sum(original_load)

aux_load = optimal_loads