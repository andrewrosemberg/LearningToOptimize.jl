using TestEnv
TestEnv.activate()

using PowerModels
using PGLib
using Gurobi
using Ipopt
using JuMP
using Juniper
using ParametricOptInterface
const POI = ParametricOptInterface
using LinearAlgebra
using Logging

# "pglib_opf_case118_ieee.m" #"pglib_opf_case5_pjm.m" # "300_ieee"
matpower_case_name = "300_ieee" 

network_data = make_basic_network(pglib(matpower_case_name))

network_formulation = DCPPowerModel # DCPPowerModel ACPPowerModel

unique([g["pmin"] for g in values(network_data["gen"])])

##########################################
## Branch and Bound
##########################################

ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0)
# optimizer = optimizer_with_attributes(Juniper.Optimizer, "nl_solver"=>ipopt)

gurobi_optimizer = Gurobi.Optimizer
jump_model = Model(gurobi_optimizer)

# Create commit variables
@variable(jump_model, u[i in 1:length(network_data["gen"])] , Bin)

# Create power flow constraints
pm = PowerModels.instantiate_model(network_data, network_formulation, PowerModels.build_opf; jump_model=jump_model)

# link generation and commit
pg = PowerModels.var(pm, :pg)
@constraint(jump_model, 
    pmin[i in 1:length(network_data["gen"])], pg[i] >= u[i] * network_data["gen"]["$i"]["pmax"] * 0.4
)

@constraint(jump_model, 
    pmax[i in 1:length(network_data["gen"])], pg[i] <= u[i] * network_data["gen"]["$i"]["pmax"]
)

JuMP.optimize!(jump_model)

termination_status(jump_model)
objective_value(jump_model)

##########################################
## Cutting Planes
##########################################

# Set the lower problem

# inner problem
poi_ipopt = () -> POI.Optimizer(Ipopt.Optimizer())
# set_optimizer_attribute(poi_ipopt, "print_level", 0)
inner_model = Model(poi_ipopt)

# Create commit parameters
start_values = zeros(length(network_data["gen"]))
@variable(inner_model, u_inner[i= 1:length(network_data["gen"])] in MOI.Parameter.(start_values))

# slack variables
@variable(inner_model, slack[i= 1:length(network_data["gen"])] >= 0)

# Create power flow constraints
pm = PowerModels.instantiate_model(network_data, network_formulation, PowerModels.build_opf; jump_model=inner_model)

# link generation and commit
pg = PowerModels.var(pm, :pg)
@constraint(inner_model, 
    pmin[i in 1:length(network_data["gen"])], pg[i] + slack[i] >= u_inner[i] * network_data["gen"]["$i"]["pmax"] * 0.4
)

@constraint(inner_model, 
    pmax[i in 1:length(network_data["gen"])], pg[i] - slack[i] <= u_inner[i] * network_data["gen"]["$i"]["pmax"]
)

obj = objective_function(inner_model)
obj = obj + sum(slack) * 1e7
set_objective(inner_model, MIN_SENSE, obj)

JuMP.optimize!(inner_model)

termination_status(inner_model)
primal_status(inner_model)
obj_value = objective_value(inner_model)
duals = [MOI.get(inner_model, POI.ParameterDual(), u_i) for u_i in u_inner]

# # cut list
# cuts_intercept = [obj_value]
# cuts_slope = [duals]
# cuts_point = [start_values]

# gurobi_optimizer = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)
# upper_model = Model(gurobi_optimizer)
# # Create commit variables
# @variable(upper_model, u[i in 1:length(network_data["gen"])] , Bin)

# # cutting planes epigraph variable
# bound = -1e7
# @variable(upper_model, θ >= bound)

# # minimize the epigraph variable
# @objective(upper_model, Min, θ)

# max_iter = 1000
# i = 1
# gap = Array{Float64}(undef, max_iter)
# upper_bound = Array{Float64}(undef, max_iter)
# lower_bound = Array{Float64}(undef, max_iter)
# while i <= max_iter
#     # Add cuts
#     @constraint(upper_model, 
#         θ >= cuts_intercept[i] + dot(cuts_slope[i], u .- cuts_point[i])
#     )

#     JuMP.optimize!(upper_model)

#     # Add point to the lists
#     if termination_status(upper_model) == MOI.OPTIMAL
#         push!(cuts_point, value.(u))
#         bound = objective_value(upper_model)
#     else
#         println("Upper problem failed")
#         break;
#     end

#     # run inner problem
#     MOI.set.(inner_model, POI.ParameterValue(), u_inner, cuts_point[i+1])
#     JuMP.optimize!(inner_model)

#     # Add cut to the lists
#     if termination_status(inner_model) == MOI.OPTIMAL || termination_status(inner_model) == MOI.LOCALLY_SOLVED
#         push!(cuts_intercept, objective_value(inner_model))
#         push!(cuts_slope, [MOI.get(inner_model, POI.ParameterDual(), u_i) for u_i in u_inner])
#     else
#         println("Inner problem failed")
#         break;
#     end

#     # test convergence
#     u_bound = minimum(cuts_intercept)
#     upper_bound[i] = u_bound
#     lower_bound[i] = bound
#     gap[i] = abs(bound - u_bound) / u_bound
#     if i > 10 && gap[i] < 0.1 && all([all(cuts_point[i] .== cuts_point[j]) for j in i-10:i-1])
#         println("Converged")
#         break;
#     else
#         @info "Iteration $i" bound cuts_intercept[i]
#     end
#     i += 1
# end
# i = ifelse(i >= max_iter, max_iter, i)

gap = gap[1:i]
upper_bound = upper_bound[1:i]
lower_bound = lower_bound[1:i]

using Plots

# Plot upper and lower bounds
plt = plot(2:i, upper_bound[2:i], label="Upper bound", title="Case $matpower_case_name", xlabel="Iteration", ylabel="Objective value");
plot!(plt, 2:i, lower_bound[2:i], label="Lower bound")


###########

# function add_deficit!(model; penalty=1e7)
#     @variable(model, deficit[1:length(model[:eq_power_balance])])
#     @variable(model, norm_deficit)
#     for (i, eq) in model[:eq_power_balance]
#         set_normalized_coefficient(eq, deficit[i], 1)
#     end
#     @constraint(model, [norm_deficit; deficit] in MOI.NormOneCone(1 + length(deficit)))
#     set_objective_coefficient(model, norm_deficit, penalty)
#     return norm_deficit
# end

using HiGHS
using JuMP
using UnitCommitment
import ParametricOptInterface as POI
using LinearAlgebra

import UnitCommitment:
    Formulation,
    KnuOstWat2018,
    MorLatRam2013,
    ShiftFactorsFormulation

# Read benchmark instance
instance = UnitCommitment.read_benchmark(
    "matpower/case118/2017-02-01",
)

inner_solver = () -> POI.Optimizer(HiGHS.Optimizer())
upper_solver = HiGHS.Optimizer

# Construct model (using state-of-the-art defaults)
model = UnitCommitment.build_model(
    instance = instance,
    optimizer = inner_solver,
)

# Solve model using cutting plane algorithm
include("src/cutting_planes.jl")

upper_model, lower_bound, upper_bound, gap = cutting_planes!(model; upper_solver, inner_solver)