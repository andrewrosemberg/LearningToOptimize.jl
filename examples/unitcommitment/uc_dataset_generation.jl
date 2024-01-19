################################################################
############## Unit Commitment Dataset Generation ##############
################################################################

using Distributed

##############
# Load Functions
##############

@everywhere include(joinpath(dirname(@__FILE__), "bnb_dataset.jl"))

@everywhere include(joinpath(dirname(dirname(@__DIR__)), "src/cutting_planes.jl"))

data_dir = joinpath(dirname(@__FILE__), "data") # joinpath(pwd(), "examples/unitcommitment", "data")

##############
# Parameters
##############
case_name = "case300"
date = "2017-01-01"
save_file = case_name * "_" * replace(date, "-" => "_") * "_h" * string(instance.time)
num_batches = 10
horizon = 2
solve_nominal = true

@info "Case: $case_name, Date: $date, Horizon: $horizon" num_batches solve_nominal

##############
# Load Instance
##############
instance = UnitCommitment.read_benchmark(
    joinpath("matpower", case_name, date),
)
instance.time = horizon

##############
# Solve and store solutions
##############

if solve_nominal
    model = build_model_uc(instance)
    uc_bnb_dataset(instance, save_file; data_dir=data_dir, model=model)
    uc_random_dataset!(instance, save_file; data_dir=data_dir, model=model)
end

# save nominal loads in a dictionary
nominal_loads = Dict()
for i in 1:length(instance.bus)
    bus = instance.bus[i]
    nominal_loads[i] = bus.load[1:horizon]
end

@distributed for _ in 1:num_batches
    # perturb loads
    uc_load_disturbances!(instance, nominal_loads)
    model = build_model_uc(instance)
    uc_bnb_dataset(instance, save_file; data_dir=data_dir, model=model)
    uc_random_dataset!(instance, save_file; data_dir=data_dir, model=model)
end
