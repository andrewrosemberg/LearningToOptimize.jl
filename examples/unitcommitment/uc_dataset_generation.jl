################################################################
############## Unit Commitment Dataset Generation ##############
################################################################

using Distributed
using Random

##############
# Load Functions
##############

@everywhere import Pkg

@everywhere Pkg.activate(dirname(dirname(@__DIR__)))

@everywhere Pkg.instantiate()

@everywhere include(joinpath(dirname(@__FILE__), "bnb_dataset.jl"))

@everywhere include(joinpath(dirname(dirname(@__DIR__)), "src/cutting_planes.jl"))

data_dir = joinpath(dirname(@__FILE__), "data")

##############
# Parameters
##############
case_name = ARGS[3] # case_name = "case300"
date = ARGS[4] # date="2017-01-01"
horizon = parse(Int, ARGS[5]) # horizon=2
save_file = case_name * "_" * replace(date, "-" => "_") * "_h" * string(horizon)
num_batches = parse(Int, ARGS[6]) # num_batches=10
solve_nominal = parse(Bool, ARGS[7]) # solve_nominal=true
num_random = parse(Int, ARGS[8]) # num_random=100
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
    if num_random > 0
        uc_random_dataset!(instance, save_file; data_dir=data_dir, model=model, num_s=num_random)
    end
end

# save nominal loads in a dictionary
nominal_loads = Dict()
for i in 1:length(instance.buses)
    bus = instance.buses[i]
    nominal_loads[i] = bus.load[1:horizon]
end

@sync @distributed for i in 1:num_batches
    rng = MersenneTwister(round(Int, i * time()))
    instance_ = deepcopy(instance)
    uc_load_disturbances!(rng, instance_, nominal_loads)
    # perturbed loads
    perturbed_loads_sum = 0.0
    for i in 1:length(instance_.buses)
        bus = instance_.buses[i]
        perturbed_loads_sum += sum(bus.load)
    end
    @info "Solving batch $i" rng perturbed_loads_sum
    model = build_model_uc(instance_)
    uc_bnb_dataset(instance_, save_file; data_dir=data_dir, model=model)
    if num_random > 0
        uc_random_dataset!(instance_, save_file; data_dir=data_dir, model=model, num_s=num_random)
    end
end

include(joinpath(dirname(@__FILE__), "compress_arrow.jl"))