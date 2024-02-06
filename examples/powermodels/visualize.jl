using Plots
using Arrow
using DataFrames
using Statistics
using LinearAlgebra

cossim(x,y) = dot(x,y) / (norm(x)*norm(y))

##############
# Parameters
##############
case_name = "pglib_opf_case300_ieee"
path_dataset = joinpath(dirname(@__FILE__), "data")
case_file_path = joinpath(path_dataset, case_name)
case_file_path_train = joinpath(case_file_path, "input", "train")
case_file_path_test = joinpath(case_file_path, "input", "test")

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

# Load input and output data tables
input_table_train = Arrow.Table(file_train)
input_table_test = Arrow.Table(file_test)

# Convert to dataframes
input_data_train = DataFrame(input_table_train)
input_data_test = DataFrame(input_table_test)

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

load_vector_train = total_load_vector(input_data_train)
load_vector_test = total_load_vector(input_data_test; is_test=true)

# Nominal Loads
nominal_loads = Vector(load_vector_train[1, Not(:id)])
norm_nominal_loads = norm(nominal_loads)

# Load divergence
theta_train = [acos(cossim(nominal_loads, Vector(load_vector_train[i, Not(:id)]))) for i in 2:size(load_vector_train, 1)] * 180 / pi
norm_sim_train = [norm(Vector(load_vector_train[i, Not(:id)])) / norm_nominal_loads for i in 2:size(load_vector_train, 1)]

theta_test = [acos(cossim(nominal_loads, Vector(load_vector_test[i, Not([:id, :in_hull])]))) for i in 1:size(load_vector_test, 1)] * 180 / pi
norm_sim_test = [norm(Vector(load_vector_test[i, Not([:id, :in_hull])])) / norm_nominal_loads for i in 1:size(load_vector_test, 1)]

# Polar Plot divergence
gr()
l = @layout [a b]
p1 = plot(scatter(theta_train, norm_sim_train, proj = :polar, label="Train", color=:blue));
p2 = plot(scatter(theta_test, norm_sim_test, proj = :polar, label="Test", color=:orange));
plot(p1, p2; title="Load Similarity", legend=:bottomright, layout = l)


