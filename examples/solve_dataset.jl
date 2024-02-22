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

problem_iterator = load(model_file, input_file, filetype)

batch_size = 10
num_problems = length(problem_iterator.ids)
num_batches = ceil(Int, num_problems / batch_size)

recorder = Recorder{filetype}(file; filterfn= (model) -> true, model=problem_iterator.model)

variable_refs = return_variablerefs(pm)
for variableref in variable_refs
    set_name(variableref, replace(name(variableref), "," => "_"))
end
set_primal_variable!(recorder, variable_refs)

ProblemIterator(ids, pairs)

########## SOLVE ##########
@sync @distributed for i in 1:num_batches
    idx_range = (i-1)*batch_size+1:min(i*batch_size, length(num_problems))
    
    @info "Batch $i of $num_batches done"
end