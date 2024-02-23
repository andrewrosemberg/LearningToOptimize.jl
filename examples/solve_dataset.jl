################################################################
###################### Dataset Generation ######################
################################################################

using Distributed
using Random

##############
# Load Functions
##############

@everywhere l2o_path = dirname(@__DIR__)

@everywhere begin
    import Pkg

    Pkg.activate(l2o_path)

    Pkg.instantiate()

    ########## SCRIPT REQUIRED PACKAGES ##########

    using L2O
    using UUIDs
    import ParametricOptInterface as POI
    using JuMP
    using UUIDs
    using Arrow

    ## SOLVER PACKAGES ##

    using Gurobi
    # using Ipopt

    POI_cached_optimizer() = Gurobi.Optimizer()

    filetype = ArrowFile
end

########## PARAMETERS ##########
model_file = joinpath(l2o_path, "examples/powermodels/data/6468_rte/input/6468_rte_SOCWRConicPowerModel_POI_load.mof.json")
input_file = joinpath(l2o_path, "examples/powermodels/data/6468_rte/input/6468_rte_POI_load_input_7f284054-d107-11ee-3fe9-09f5e129b1ad")

save_path = joinpath(l2o_path, "examples/powermodels/data/6468_rte/output/")
case_name = split(split(model_file, ".mof.")[1], "/")[end]
processed_output_files = [file for file in readdir(save_path; join=true) if occursin(case_name, file)]
ids = Vector(Arrow.Table(processed_output_files).id)
batch_size = 20

########## SOLVE ##########

problem_iterator_factory, num_batches = load(model_file, input_file, filetype; batch_size=batch_size, ignore_ids=ids)

@sync @distributed for i in 1:num_batches
    problem_iterator = problem_iterator_factory(i)
    set_optimizer(problem_iterator.model, () -> POI_cached_optimizer())
    output_file = joinpath(save_path, "$(case_name)_output_$(uuid1())")
    recorder = Recorder{filetype}(output_file; filterfn= (model) -> true, model=problem_iterator.model)
    successfull_solves = solve_batch(problem_iterator, recorder)
    @info "Solved $(length(successfull_solves)) problems"
end
