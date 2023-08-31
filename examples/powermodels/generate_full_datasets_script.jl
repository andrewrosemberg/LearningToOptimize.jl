using TestEnv
TestEnv.activate()

using Arrow
using L2O
using Test
using UUIDs
using PowerModels
using Clarabel
using JuMP.MOI
import ParametricOptInterface as POI

cached = MOI.Bridges.full_bridge_optimizer(
    MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        Clarabel.Optimizer(),
    ),
    Float64,
)

# Paths
path_powermodels = joinpath(pwd(), "examples", "powermodels")
path = joinpath(path_powermodels, "data")
include(joinpath(path_powermodels, "pglib_datagen.jl"))

# Parameters
num_batches = 1
num_p = 10
filetype = ArrowFile

# Case name
case_name = "pglib_opf_case300_ieee"
network_formulation = SOCWRConicPowerModel
case_file_path = joinpath(path, case_name)
solver = () -> POI.Optimizer(cached)

# Generate dataset
success_solves = 0.0
for i in 1:num_batches
    _success_solves, number_variables, number_loads, batch_id = generate_dataset_pglib(case_file_path, case_name; 
        num_p=num_p, filetype=filetype, network_formulation=network_formulation, solver=solver,
    )
    success_solves += _success_solves
end
success_solves /= num_batches
