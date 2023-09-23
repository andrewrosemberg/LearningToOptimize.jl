# run with: julia ./examples/powermodels/generate_full_datasets_script.jl 
using TestEnv
TestEnv.activate()

using Arrow
using L2O
using Test
using UUIDs
using PowerModels
using Clarabel
import JuMP.MOI as MOI
import ParametricOptInterface as POI
using Gurobi

using NonconvexNLopt

# using QuadraticToBinary

########## SOLVERS ##########

cached = () -> MOI.Bridges.full_bridge_optimizer(
    MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        Clarabel.Optimizer(),
    ),
    Float64,
)

POI_cached_optimizer() = POI.Optimizer(cached())

########## DATASET GENERATION ##########

# Paths
path_powermodels = joinpath(pwd(), "examples", "powermodels")
path = joinpath(path_powermodels, "data")
include(joinpath(path_powermodels, "pglib_datagen.jl"))

# Parameters
filetype = CSVFile # CSVFile # ArrowFile

# Case name
case_name = "pglib_opf_case300_ieee" # pglib_opf_case300_ieee # pglib_opf_case5_pjm
network_formulation = SOCWRConicPowerModel # SOCWRConicPowerModel # DCPPowerModel
case_file_path = joinpath(path, case_name)
mkpath(case_file_path)

# Generate dataset
# num_batches = 1
# num_p = 200
# global success_solves = 0.0
# for i in 1:num_batches
#     _success_solves, number_variables, number_loads, batch_id = generate_dataset_pglib(case_file_path, case_name; 
#         num_p=num_p, filetype=filetype, network_formulation=network_formulation, optimizer=POI_cached_optimizer,
#         internal_load_sampler= (_o, n) -> load_sampler(_o, n, max_multiplier=1.25, min_multiplier=0.8, step_multiplier=0.01)
#     )
#     global success_solves += _success_solves
# end
# success_solves /= num_batches

# @info "Success solves Normal: $(success_solves)"

# Generate Line-search dataset

matpower_case_name = case_name * ".m"
network_data = make_basic_network(pglib(matpower_case_name))

early_stop_fn = (model, status, recorder) -> !status
step_multiplier = 1.01
num_loads = length(network_data["load"])
num_batches = num_loads * 2 + 1
num_p = 10

function line_sampler(_o, n, idx, num_inputs, ibatc)
    if (idx == ibatc) || (ibatc == num_inputs + 1)
        return [_o * step_multiplier ^ (j-1) for j in 1:n]
    else
        return ones(n)
    end
end

global success_solves = 0.0
global batch_id = string(uuid1())
for ibatc in 1:num_batches
    _success_solves, number_variables, number_loads, b_id = generate_dataset_pglib(case_file_path, case_name; 
        num_p=num_p, filetype=filetype, network_formulation=network_formulation, optimizer=POI_cached_optimizer,
        internal_load_sampler= (_o, n, idx, num_inputs) -> line_sampler(_o, n, idx, num_inputs, ibatc),
        early_stop_fn=early_stop_fn,
        batch_id=batch_id,
    )
    global success_solves += _success_solves
end
success_solves /= num_batches

@info "Success solves: $(success_solves * 100) % of $(num_batches * num_p)"

# Generate worst case dataset

# function optimizer_factory()
#     IPO_OPT = Gurobi.Optimizer() # MadNLP.Optimizer(print_level=MadNLP.INFO, max_iter=100)
#     # IPO = MOI.Bridges.Constraint.SOCtoNonConvexQuad{Float64}(IPO_OPT)
#     # MIP = QuadraticToBinary.Optimizer{Float64}(IPO)
#     return () -> IPO_OPT
# end

# global success_solves = 0.0
# for i in 1:num_batches
#     _success_solves, number_variables, number_loads, batch_id = generate_worst_case_dataset(case_file_path, case_name; 
#         num_p=num_p, filetype=filetype, network_formulation=network_formulation, optimizer_factory=optimizer_factory,
#         hook = (model) -> set_optimizer_attribute(model, "NonConvex", 2)
#     )
#     global success_solves += _success_solves
# end
# success_solves /= num_batches

# @info "Success solves Worst Case: $(success_solves) of $(num_batches * num_p)"

# global success_solves = 0.0
# for i in 1:num_batches
#     _success_solves, number_variables, number_loads, batch_id = generate_worst_case_dataset_Nonconvex(case_file_path, case_name; 
#         num_p=num_p, filetype=filetype, network_formulation=network_formulation, optimizer=POI_cached_optimizer,
#     )
#     global success_solves += _success_solves
# end
# success_solves /= num_batches

# @info "Success solves Worst Case: $(success_solves * 100) of $(num_batches * num_p)"
