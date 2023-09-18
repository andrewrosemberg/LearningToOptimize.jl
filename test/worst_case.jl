using PowerModels
using Ipopt
using PGLib
using JuMP
using Dualization
import JuMP.MOI as MOI

function _moi_set_upper_bound(
    moi_backend,
    idx::MOI.VariableIndex,
    upper,
)
    new_set = MOI.LessThan(upper)

    JuMP._moi_add_constraint(moi_backend, idx, new_set)

    return
end

function _moi_set_lower_bound(
    moi_backend,
    idx::MOI.VariableIndex,
    lower,
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
load_moi_idx = MOI.VariableIndex[i.index for i in load_var]

# Instantiate the model
pm = instantiate_model(
    network_data,
    network_formulation,
    PowerModels.build_opf;
    setting=Dict("output" => Dict("duals" => true)),
    jump_model=model,
)

# Solve the model
# JuMP.optimize!(model)

# optimal_cost = JuMP.objective_value(model)

# Dualize the model
dual_st = Dualization.dualize(JuMP.backend(model), 
    variable_parameters = load_moi_idx
)

dual_model = dual_st.dual_model
primal_dual_map = dual_st.primal_dual_map

# for (i,l) in enumerate(load_var)
#     _moi_set_upper_bound(dual_model, primal_dual_map.primal_parameter[l.index], max_total_load)
#     _moi_set_lower_bound(dual_model, primal_dual_map.primal_parameter[l.index], 0.0)
# end

jump_dual_model = JuMP.Model()
map_moi_to_jump = MOI.copy_to(JuMP.backend(jump_dual_model), dual_model)
set_optimizer(jump_dual_model, solver)

load_dual_idxs = [map_moi_to_jump[primal_dual_map.primal_parameter[l]].value for l in load_moi_idx]

load_var_dual = JuMP.all_variables(jump_dual_model)[load_dual_idxs]

for (i,l) in enumerate(load_var_dual)
    @constraint(jump_dual_model, l <= original_load[i]*1.5)
    @constraint(jump_dual_model, l >= 0.0)
end
for (i, l) in enumerate(values(network_data["load"]))
    l["pd"] = load_var_dual[i]
end

obj = objective_function(jump_dual_model)

pm = instantiate_model(
    network_data,
    network_formulation,
    PowerModels.build_opf;
    setting=Dict("output" => Dict("duals" => true)),
    jump_model=jump_dual_model,
)

@objective(jump_dual_model, Max, obj)

JuMP.optimize!(jump_dual_model)

optimal_dual_cost = JuMP.objective_value(jump_dual_model)

optimal_loads = value.(load_var_dual)

cost_diff = (optimal_dual_cost - optimal_cost) / optimal_cost

load_diff = sum(abs.(optimal_loads - original_load)) ./ sum(original_load)