using JuMP
using PowerModels
using PGLib
using Gurobi

optimizer = Gurobi.Optimizer
network_formulation = ACPPowerModel # ACPPowerModel SOCWRConicPowerModel DCPPowerModel

matpower_case_name = "6468_rte"

network_data = make_basic_network(pglib(matpower_case_name))

# The problem to iterate over
model = JuMP.Model(optimizer)

# Save original load value and Link POI
num_loads = length(network_data["load"])

@variable(model, load_scaler[i=1:num_loads] in MOI.Parameter.(1.0))

for (str_i, l) in network_data["load"]
    i = parse(Int, str_i)
    l["pd"] = load_scaler[i] * l["pd"]
    l["qd"] = load_scaler[i] * l["qd"]
end

pm = instantiate_model(
    network_data,
    network_formulation,
    PowerModels.build_opf;
    setting=Dict("output" => Dict("branch_flows" => true, "duals" => true)),
    jump_model=model,
)

write_to_file(model, "$(matpower_case_name)_$(network_formulation)_POI_load.mof.json")

# dest_model = read_from_file("$(matpower_case_name)_$(network_formulation)_POI_load.mof.json")