# run with: julia ./examples/powermodels/generate_full_datasets_script.jl "./examples/powermodels/data/pglib_opf_case300_ieee/case300.config.toml"
config_path = ARGS[1]

import Pkg; Pkg.activate(".")

using TestEnv
TestEnv.activate()

########## SCRIPT REQUIRED PACKAGES ##########

using L2O
using Arrow
using Test
using UUIDs
using PowerModels
import JuMP.MOI as MOI
import ParametricOptInterface as POI
using TOML

PowerModels.silence()

## SOLVER PACKAGES ##

using Clarabel
using Gurobi
using NonconvexNLopt

########## POI SOLVER ##########

cached = () -> MOI.Bridges.full_bridge_optimizer(
    MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        Clarabel.Optimizer(),
    ),
    Float64,
)

POI_cached_optimizer() = POI.Optimizer(cached())

########## PARAMETERS ##########

config = TOML.parsefile(config_path)
path = config["export_dir"]

path_powermodels = joinpath(dirname(@__FILE__)) # TODO: Make it a submodule
include(joinpath(path_powermodels, "pglib_datagen.jl"))

filetype = ArrowFile # ArrowFile # CSVFile

case_name = config["case_name"]
case_file_path = joinpath(path, case_name)
mkpath(case_file_path)
network_formulation= eval(Symbol(ARGS[2])) # SOCWRConicPowerModel # DCPPowerModel

########## SAMPLER DATASET GENERATION ##########

if haskey(config, "sampler")
    num_batches = config["sampler"]["num_batches"]
    num_p = config["sampler"]["num_samples"]
    global success_solves = 0.0
    for i in 1:num_batches
        _success_solves, number_variables, number_loads, batch_id = generate_dataset_pglib(case_file_path, case_name; 
            num_p=num_p, filetype=filetype, network_formulation=network_formulation, optimizer=POI_cached_optimizer,
            internal_load_sampler= (_o, n) -> load_sampler(_o, n, max_multiplier=1.25, min_multiplier=0.8, step_multiplier=0.01)
        )
        global success_solves += _success_solves
    end
    success_solves /= num_batches

    @info "Success solves Normal: $(success_solves)"
end

########## LINE SEARCH DATASET GENERATION ##########

if haskey(config, "line_search")
    network_data = make_basic_network(pglib(case_name * ".m"))
    step_multiplier = 1.01
    num_loads = length(network_data["load"])
    num_batches = num_loads + 1
    num_p = config["line_search"]["num_samples"]
    early_stop_fn = (model, status, recorder) -> !status

    global success_solves = 0.0
    for ibatc in 1:num_batches
        _success_solves, number_variables, number_loads, b_id = generate_dataset_pglib(case_file_path, case_name; 
            num_p=num_p, filetype=filetype, network_formulation=network_formulation, optimizer=POI_cached_optimizer,
            internal_load_sampler= (_o, n, idx, num_inputs) -> line_sampler(_o, n, idx, num_inputs, ibatc; step_multiplier=step_multiplier),
            early_stop_fn=early_stop_fn,
        )
        global success_solves += _success_solves
    end
    success_solves /= num_batches
    
    @info "Success solves: $(success_solves * 100) % of $(num_batches * num_p)"
end

########## WORST CASE DUAL DATASET GENERATION ##########
if haskey(config, "worst_case_dual")
    num_p = config["worst_case_dual"]["num_samples"]
    function optimizer_factory()
        IPO_OPT = Gurobi.Optimizer() 
        # IPO_OPT = MadNLP.Optimizer(print_level=MadNLP.INFO, max_iter=100)
        # IPO = MOI.Bridges.Constraint.SOCtoNonConvexQuad{Float64}(IPO_OPT)
        # MIP = QuadraticToBinary.Optimizer{Float64}(IPO)
        return () -> IPO_OPT
    end

    success_solves, number_variables, number_loads, batch_id = generate_worst_case_dataset(case_file_path, case_name; 
        num_p=num_p, filetype=filetype, network_formulation=network_formulation, optimizer_factory=optimizer_factory,
        hook = (model) -> set_optimizer_attribute(model, "NonConvex", 2)
    )

    @info "Success solves Worst Case: $(success_solves) of $(num_p)"
end

########## WORST CASE NONCONVEX DATASET GENERATION ##########
if haskey(config, "worst_case_nonconvex")
    num_p = config["worst_case_nonconvex"]["num_samples"]

    success_solves, number_variables, number_loads, batch_id = generate_worst_case_dataset_Nonconvex(case_file_path, case_name; 
        num_p=num_p, filetype=filetype, network_formulation=network_formulation, optimizer=POI_cached_optimizer,
    )

    @info "Success solves Worst Case: $(success_solves * 100) of $(num_p)"
end
