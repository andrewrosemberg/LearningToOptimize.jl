using JuMP
using PowerModels
using PGLib
using Clarabel
using Ipopt
import ParametricOptInterface as POI

cached_clara =
    () -> MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            Clarabel.Optimizer(),
        ),
        Float64,
    )

POI_cached_optimizer_clara() = POI.Optimizer(cached_clara())

ipopt = Ipopt.Optimizer()
MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
cached =
    () -> MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            ipopt,
        ),
        Float64,
    )
POI_cached_optimizer() = POI.Optimizer(cached())

network_formulation = ACPPowerModel # ACPPowerModel SOCWRConicPowerModel DCPPowerModel

matpower_case_name = "6468_rte"

network_data = make_basic_network(pglib(matpower_case_name))

# The problem to iterate over
model = JuMP.Model(() -> POI_cached_optimizer())

# Save original load value and Link POI
num_loads = length(network_data["load"])

@variable(model, load_scaler[i = 1:num_loads] in MOI.Parameter.(1.0)) # needs fixing -> need to be instantiated after all other variables

for (str_i, l) in network_data["load"]
    i = parse(Int, str_i)
    l["pd"] = load_scaler[i] * l["pd"]
    l["qd"] = load_scaler[i] * l["qd"]
end

pm = instantiate_model(
    network_data,
    network_formulation,
    PowerModels.build_opf;
    setting = Dict("output" => Dict("branch_flows" => true, "duals" => true)),
    jump_model = model,
)

JuMP.optimize!(model)
JuMP.termination_status(model)
JuMP.objective_value(model)

write_to_file(model, "$(matpower_case_name)_$(network_formulation)_POI_load.mof.json")

# dest_model = read_from_file("examples/powermodels/data/$(matpower_case_name)/input/$(matpower_case_name)_$(network_formulation)_POI_load.mof.json")
# set_optimizer(dest_model, () -> POI_cached_optimizer())
# JuMP.optimize!(dest_model)
# JuMP.termination_status(dest_model)
# JuMP.objective_value(dest_model)
