####################################################
############## PowerModels Data Split ##############
####################################################
import Pkg
Pkg.activate(dirname(dirname(@__DIR__)))

using Distributed
using Random
using Arrow
using CSV
using MLUtils

##############
# Load Packages everywhere
##############

@everywhere import Pkg
@everywhere Pkg.activate(dirname(dirname(@__DIR__)))
@everywhere Pkg.instantiate()
@everywhere using DataFrames
@everywhere using L2O
@everywhere using Gurobi

# Paths
case_name = "pglib_opf_case300_ieee" # pglib_opf_case300_ieee # pglib_opf_case5_pjm
filetype = ArrowFile # ArrowFile # CSVFile
path_dataset = joinpath(dirname(@__FILE__), "data")
case_file_path = joinpath(path_dataset, case_name)
case_file_path_input = joinpath(case_file_path, "input")

# Load input and output data tables
iter_files_in = readdir(joinpath(case_file_path_input))
iter_files_in = filter(x -> occursin(string(filetype), x), iter_files_in)
file_ins = [
    joinpath(case_file_path_input, file) for file in iter_files_in if occursin("input", file)
]
batch_ids = [split(split(file, "_")[end], ".")[1] for file in file_ins]

# Load input and output data tables
if filetype === ArrowFile
    input_table_train = Arrow.Table(file_ins)
else
    input_table_train = CSV.read(file_ins[train_idx], DataFrame)
end

# Convert to dataframes
input_data = DataFrame(input_table_train)

# Split the data into training and test sets
# seed random number generator
Random.seed!(123)
train_idx, test_idx = splitobs(1:size(input_data, 1), at=(0.7), shuffle=true)

train_table = input_data[train_idx, :]
test_table = input_data[test_idx, :]

batch_size = 50

num_batches = ceil(Int, length(test_idx) / batch_size)

inhull = Array{Bool}(undef, length(test_idx))
@sync @distributed for i in 1:num_batches
    idx_range = (i-1)*batch_size+1:min(i*batch_size, length(test_idx))
    batch = test_table[idx_range, :]
    inhull[idx_range] = inconvexhull(Matrix(train_table[!, Not(:id)]), Matrix(batch[!, Not(:id)]), Gurobi.Optimizer)
end

test_table.in_train_convex_hull = inhull

# Save the training and test sets
mkpath(joinpath(case_file_path_input, "train"))
mkpath(joinpath(case_file_path_input, "test"))

if filetype === ArrowFile
    Arrow.write(Arrow.Table(train_table), joinpath(case_file_path_input, "train", case_name * "_train_input" * ".arrow"))
    Arrow.write(Arrow.Table(test_table), joinpath(case_file_path_input, "test", case_name * "_test_input" * ".arrow"))
else
    CSV.write(joinpath(case_file_path_input, "train", case_name * "_train_input" * ".csv"), train_table)
    CSV.write(joinpath(case_file_path_input, "test", case_name * "_test_input" * ".csv"), test_table)
end