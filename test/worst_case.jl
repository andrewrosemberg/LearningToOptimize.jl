using PowerModels
using Ipopt
using PGLib
using JuMP
using Dualization
import JuMP.MOI as MOI

# Load data
case_name = "pglib_opf_case5_pjm"
matpower_case_name = case_name * ".m"
network_data_original = pglib(matpower_case_name)
network_data = deepcopy(network_data_original)

# Case Parameters
network_formulation=DCPPowerModel
original_load = [l["pd"] for l in values(network_data["load"])]
max_total_load = sum(original_load) * 1.1

# Create primal model
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

# Dualize the model
dual_st = Dualization.dualize(JuMP.backend(model), 
    variable_parameters = load_moi_idx
)

dual_model = dual_st.dual_model
primal_dual_map = dual_st.primal_dual_map

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

load_diff = sum(abs.(optimal_loads - original_load)) ./ sum(original_load)

# Create final primal model
network_data = deepcopy(network_data_original)
model = Model(solver)
for (i, l) in enumerate(values(network_data["load"]))
    l["pd"] = optimal_loads[i]
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

# Check the solution
optimal_final_cost = JuMP.objective_value(model)
termination_status = JuMP.termination_status(model)
solution_primal_status = JuMP.primal_status(model)
solution_dual_status = JuMP.dual_status(model)
termination_status == MOI.INFEASIBLE && @error("Optimal solution not found")
solution_primal_status != MOI.FEASIBLE_POINT && @error("Primal solution not found")
solution_dual_status != MOI.FEASIBLE_POINT && @error("Dual solution not found")
isapprox(optimal_final_cost, optimal_dual_cost; rtol=1e-4) || @warn("Final cost is not equal to dual cost")
