################################################################
############## Unit Commitment Dataset Generation ##############
################################################################

using Distributed

##############
# Load Functions
##############

@everywhere import Pkg

@everywhere Pkg.activate(dirname(dirname(@__DIR__)))

@everywhere Pkg.instantiate()

@everywhere include(joinpath(dirname(@__FILE__), "bnb_dataset.jl"))

@everywhere include(joinpath(dirname(dirname(@__DIR__)), "src/cutting_planes.jl"))

data_dir = joinpath(dirname(@__FILE__), "data") # joinpath(pwd(), "examples/unitcommitment", "data")

##############
# Parameters
##############
case_name = ARGS[3] #"case300"
date = ARGS[4] # "2017-01-01"
horizon = parse(Int, ARGS[5]) # 2
save_file = case_name * "_" * replace(date, "-" => "_") * "_h" * string(horizon)
num_batches = parse(Int, ARGS[6]) # 10
solve_nominal = parse(Bool, ARGS[7]) #true
data_dir = joinpath(data_dir, case_name, date, "h" * string(horizon))
mkpath(joinpath(data_dir, "input"))
mkpath(joinpath(data_dir, "output"))

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
for i in 1:length(instance.buses)
    bus = instance.buses[i]
    nominal_loads[i] = bus.load[1:horizon]
end

@distributed for _ in 1:num_batches
    # perturb loads
    uc_load_disturbances!(instance, nominal_loads)
    model = build_model_uc(instance)
    uc_bnb_dataset(instance, save_file; data_dir=data_dir, model=model)
    uc_random_dataset!(instance, save_file; data_dir=data_dir, model=model)
end

