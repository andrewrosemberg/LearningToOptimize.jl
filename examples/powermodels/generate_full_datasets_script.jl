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
num_batches = 1
num_p = 200
filetype = ArrowFile

# Case name
case_name = "pglib_opf_case300_ieee" # "pglib_opf_case300_ieee"
network_formulation = SOCWRConicPowerModel # SOCWRConicPowerModel # DCPPowerModel
case_file_path = joinpath(path, case_name)
mkpath(case_file_path)

# Generate dataset
# global success_solves = 0.0
# for i in 1:num_batches
#     _success_solves, number_variables, number_loads, batch_id = generate_dataset_pglib(case_file_path, case_name; 
#         num_p=num_p, filetype=filetype, network_formulation=network_formulation, optimizer=POI_cached_optimizer,
#         load_sampler= (_o, n) -> load_sampler(_o, n, max_multiplier=1.25, min_multiplier=0.8, step_multiplier=0.01)
#     )
#     global success_solves += _success_solves
# end
# success_solves /= num_batches

# @info "Success solves Normal: $(success_solves)"

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

global success_solves = 0.0
for i in 1:num_batches
    _success_solves, number_variables, number_loads, batch_id = generate_worst_case_dataset_Nonconvex(case_file_path, case_name; 
        num_p=num_p, filetype=filetype, network_formulation=network_formulation, optimizer=POI_cached_optimizer,
    )
    global success_solves += _success_solves
end
success_solves /= num_batches

@info "Success solves Worst Case: $(success_solves * 100) of $(num_batches * num_p)"