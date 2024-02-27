using Plots
using Arrow
using DataFrames
using Statistics
using LinearAlgebra

cossim(x,y) = dot(x,y) / (norm(x)*norm(y))

##############
# Parameters
##############
network_formulation = "ACPPowerModel" # ACPPowerModel "DCPPowerModel" # "SOCWRConicPowerModel"
case_name = "6468_rte" # pglib_opf_case300_ieee # 6468_rte 
path_dataset = joinpath(dirname(@__FILE__), "data")
case_file_path = joinpath(path_dataset, case_name)
case_file_path_train = joinpath(case_file_path, "input", "train")
case_file_path_test = joinpath(case_file_path, "input", "test")
case_file_path_output = joinpath(case_file_path, "output", string(network_formulation))

##############
# Load Data
##############
iter_files_train = readdir(joinpath(case_file_path_train))
iter_files_test = readdir(joinpath(case_file_path_test))
iter_files_train = filter(x -> occursin("arrow", x), iter_files_train)
iter_files_test = filter(x -> occursin("arrow", x), iter_files_test)
file_train = [
    joinpath(case_file_path_train, file) for file in iter_files_train if occursin("input", file)
]
file_test = [
    joinpath(case_file_path_test, file) for file in iter_files_test if occursin("input", file)
]
iter_files_out = readdir(joinpath(case_file_path_output))
iter_files_out = filter(x -> occursin("arrow", x), iter_files_out)
file_outs = [
    joinpath(case_file_path_output, file) for file in iter_files_out if occursin("output", file)
]

# Load input and output data tables
input_table_train = Arrow.Table(file_train)
input_table_test = Arrow.Table(file_test)
output_table = Arrow.Table(file_outs)

# Convert to dataframes
input_data_train = DataFrame(input_table_train)
input_data_test = DataFrame(input_table_test)
output_data = DataFrame(output_table)
output_data.operational_cost = output_data.objective
output_data = output_data[output_data.objective .> 10, :]
input_data = vcat(input_data_train, input_data_test[!, Not(:in_train_convex_hull)])

##############
# SOC VS AC
##############
network_formulation_soc = "SOCWRConicPowerModel"
case_file_path_output_soc = joinpath(case_file_path, "output", string(network_formulation_soc))
iter_files_out_soc = readdir(case_file_path_output_soc)
iter_files_out_soc = filter(x -> occursin("arrow", x), iter_files_out_soc)
file_outs_soc = [
    joinpath(case_file_path_output_soc, file) for file in iter_files_out_soc if occursin("output", file)
]
output_table_soc = Arrow.Table(file_outs_soc)
output_data_soc = DataFrame(output_table_soc)
output_data_soc.operational_cost_soc = output_data_soc.objective
output_data_soc = output_data_soc[output_data_soc.objective .> 10, :]

# compare SOC and AC operational_cost by id
ac_soc = innerjoin(output_data[!, [:id, :operational_cost]], output_data_soc[!, [:id, :operational_cost_soc]], on=:id, makeunique=true)

ac_soc.error = (ac_soc.operational_cost .- ac_soc.operational_cost_soc) ./ ac_soc.operational_cost * 100
mean(ac_soc.error)
maximum(ac_soc.error)
ac_soc[findmax(ac_soc.error)[2], :]
##############
# Plots
##############

# Load vectors
function total_load_vector(input_data; is_test=false)
    df = DataFrame()
    df.id = input_data.id
    if is_test
        df.in_hull = input_data.in_train_convex_hull
    end
    num_loads = length([col for col in names(input_data) if occursin("pd", col)])
    for i in 1:num_loads
        df[!, "load[$i]"] = zeros(size(input_data, 1))
    end
    for j in 1:size(input_data, 1), i in 1:num_loads
        df[j, "load[$i]"] = sqrt(input_data[j, "pd[$i]"]^2 + input_data[j, "qd[$i]"]^2)
    end
    return df
end

function total_load_vector_annon(input_data)
    df = DataFrame()
    df.id = input_data.id
    num_loads = floor(Int, length(names(input_data[!, Not(:id)])) / 2)
    for i in 1:num_loads
        df[!, "load[$i]"] = zeros(size(input_data, 1))
    end
    for j in 1:size(input_data, 1), i in 1:num_loads
        df[j, "load[$i]"] = sqrt(input_data[j, i+1]^2 + input_data[j, i+num_loads+1]^2)
    end
    return df
end

######### Plot Load Vectors #########
load_vector_train = total_load_vector(input_data_train)
load_vector_test = total_load_vector(input_data_test; is_test=true)

# Nominal Loads
nominal_loads = Vector(input_data_train[1, Not(:id)])
norm_nominal_loads = norm(nominal_loads)

# Load divergence
theta_train = [acos(cossim(nominal_loads, Vector(input_data_train[i, Not(:id)]))) for i in 2:10000] * 180 / pi
norm_sim_train = [norm(Vector(input_data_train[i, Not(:id)])) / norm_nominal_loads for i in 2:10000]

theta_test = [acos(cossim(nominal_loads, Vector(load_vector_test[i, Not([:id, :in_hull])]))) for i in 1:size(load_vector_test, 1)] * 180 / pi
norm_sim_test = [norm(Vector(load_vector_test[i, Not([:id, :in_hull])])) / norm_nominal_loads for i in 1:size(load_vector_test, 1)]

# Polar Plot divergence
gr()
l = @layout [a b]
p1 = plot(scatter(theta_train, norm_sim_train, proj = :polar, label="Train", color=:blue));
p2 = plot(scatter(theta_test, norm_sim_test, proj = :polar, label="Test", color=:orange));
plot(p1, p2; title="Load Similarity", legend=:bottomright, layout = l)

######### Plot Objective Function #########
# Plot objective function for a single direction

# Select two points in the extremes
load_vector = total_load_vector(input_data)

nominal_loads = Vector(load_vector[1, Not(:id)])
norm_nominal_loads = norm(nominal_loads)

load_vector.norm_loads = [norm(Vector(load_vector[i, Not(:id)])) for i in 1:size(load_vector, 1)] ./ norm_nominal_loads
# join input and output data
joined_data = innerjoin(load_vector, output_data[!, [:id, :operational_cost]], on=:id)

# get k extreme points
using L2O
using Gurobi
function maxk(a, k)
    b = partialsortperm(a, 1:k, rev=true)
    return collect(zip(b, a[b]))
end
function mink(a, k)
    b = partialsortperm(a, 1:k, rev=false)
    return collect(zip(b, a[b]))
end
k = 10
idx_maxs = maxk(joined_data.norm_loads, k)
idx_mins = mink(joined_data.norm_loads, k)

# Objective function
instances_in_convex_hulls = Array{DataFrame}(undef, k)
for i in 1:k
    @info "Processing instance $i"
    instance_load_max = joined_data[idx_maxs[i][1]:idx_maxs[i][1], :]
    instance_load_min = joined_data[idx_mins[i][1]:idx_mins[i][1], :]

    # find instances between the two points (i.e. in their convex hull)
    in_convex_hull = inconvexhull(
        Matrix(vcat(instance_load_max, 
        instance_load_min)[!, Not([:id, :norm_loads, :operational_cost])]), Matrix(joined_data[!, Not([:id, :norm_loads, :operational_cost])]),
        Gurobi.Optimizer, silent=false, tol=0.1 # close to convex hull
    )

    instances_in_convex_hulls[i] = sort(joined_data[in_convex_hull, [:operational_cost, :norm_loads]], :norm_loads)
end

# Plot
plotly()

plt = plot(color=:blue, legend=false, xlabel="Total Load (%)", ylabel="Operational Cost", title="AC OPF - IEEE300");
for i in 1:k
    plot!(instances_in_convex_hulls[i][!, :norm_loads] * 100, instances_in_convex_hulls[i][!, :operational_cost], label="Direction $i", color=:red, alpha=0.4)
end
plt
