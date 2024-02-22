################################################################
###################### Dataset Generation ######################
################################################################

using Distributed
using Random

##############
# Load Functions
##############

@everywhere import Pkg

@everywhere Pkg.activate(dirname(dirname(@__DIR__)))

@everywhere Pkg.instantiate()

########## SCRIPT REQUIRED PACKAGES ##########

@everywhere using L2O
@everywhere using UUIDs
@everywhere import ParametricOptInterface as POI

## SOLVER PACKAGES ##

@everywhere using Gurobi
@everywhere using Ipopt

POI_cached_optimizer() = Gurobi.Optimizer()

########## PARAMETERS ##########
filetype = ArrowFile
model_file = "examples/powermodels/data/6468_rte/input/6468_rte_SOCWRConicPowerModel_POI_load.mof.json"
input_file = "examples/powermodels/data/6468_rte/input/6468_rte_POI_load_input_7f284054-d107-11ee-3fe9-09f5e129b1ad.arrow"

save_path = "examples/powermodels/data/6468_rte/output/"
case_name = split(split(model_file, ".mof.")[1], "/")[end]
batch_size = 200

########## SOLVE ##########

problem_iterators = load(model_file, input_file, filetype; batch_size=batch_size)

@sync @distributed for problem_iterators in problem_iterators
    set_optimizer(problem_iterator.model, POI_cached_optimizer())
    output_file = joinpath(save_path, "$(case_name)_output_$(UUID())")
    recorder = Recorder{filetype}(output_file; filterfn= (model) -> true, model=problem_iterator.model)
    successfull_solves = solve_batch(problem_iterator, recorder)
    @info "Solved $(length(successfull_solves)) problems"
end
