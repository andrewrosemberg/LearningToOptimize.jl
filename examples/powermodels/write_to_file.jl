using JuMP
using PowerModels
using PGLib
using Gurobi

optimizer = Gurobi.Optimizer
network_formulation = SOCWRConicPowerModel

matpower_case_name = "6468_rte"

network_data = make_basic_network(pglib(matpower_case_name))

# The problem to iterate over
model = JuMP.Model(optimizer)

# Save original load value and Link POI
num_loads = length(network_data["load"])
num_inputs = num_loads * 2
original_load = vcat(
    [network_data["load"]["$l"]["pd"] for l in 1:num_loads],
    [network_data["load"]["$l"]["qd"] for l in 1:num_loads],
)

p = load_parameter_factory(model, 1:num_inputs; load_set=MOI.Parameter.(original_load))

for (str_i, l) in network_data["load"]
    i = parse(Int, str_i)
    l["pd"] = p[i]
    l["qd"] = p[num_loads + i]
end

pm = instantiate_model(
    network_data,
    network_formulation,
    PowerModels.build_opf;
    setting=Dict("output" => Dict("branch_flows" => true, "duals" => true)),
    jump_model=model,
)

write_to_file(model, "$(matpower_case_name)_$(network_formulation)_POI_load.mof.json")

dest_model = read_from_file("$(matpower_case_name)_$(network_formulation)_POI_load.mof.json")