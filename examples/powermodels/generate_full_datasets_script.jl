using TestEnv
TestEnv.activate()

using Arrow
using L2O
using Test
using UUIDs
using PowerModels

# Paths
path_powermodels = joinpath(pwd(), "examples", "powermodels")
path = joinpath(path_powermodels, "data")
include(joinpath(path_powermodels, "pglib_datagen.jl"))

# Parameters
num_batches = 2
num_p = 1000
filetype = ArrowFile

# Case name
case_name = "pglib_opf_case5_pjm"
network_formulation = DCPPowerModel
case_file_path = joinpath(path, case_name)

# Generate dataset
success_solves = 0.0
for i in 1:num_batches
    _success_solves, number_variables, number_loads, batch_id = generate_dataset_pglib(case_file_path, case_name; 
        num_p=num_p, filetype=filetype, network_formulation=network_formulation
    )
    success_solves += _success_solves
end
success_solves /= num_batches
