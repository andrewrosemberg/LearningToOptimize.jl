using Arrow
using DataFrames

# Data Parameters
case_name = "case300"
date = "2017-01-01"
horizon = 2
path_dataset = joinpath(
    dirname(@__FILE__), "data", case_name, date, "h" * string(horizon)
)
case_file_path = path_dataset # joinpath(path_dataset, case_name)

# Load input and output data tables
iter_files = readdir(case_file_path)
iter_files = [
    joinpath(case_file_path, file) for file in iter_files if occursin(case_name, file)
]
file_ins = [
    joinpath(case_file_path, file) for file in iter_files if occursin("input", file)
]

batch_ids = [split(split(file, "_")[end], ".")[1] for file in file_ins]

@info "Compressing files in $(case_file_path)" batch_ids

file_name = split(split(file_ins[1], "_input")[1], "/")[end]

# move input files to input folder
for file in file_ins
    mv(file, joinpath(case_file_path, "input"))
end

# compress output files per batch id
iter_files = readdir(joinpath(case_file_path))
for batch_id in batch_ids
    file_outs = [
        joinpath(case_file_path, file)
        for file in iter_files
        if occursin("output", file) && occursin(batch_id, file) && occursin("arrow", file)
    ]
    if length(file_outs) == 1
        continue
    end

    # Load input and output data tables
    output_table = Arrow.Table(file_outs)

    # Save compressed files
    Arrow.write(
        joinpath(case_file_path, "output", "$(file_name)_output_" * batch_id * ".arrow"),
        output_table,
    )

    # Delete uncompressed files
    for file in file_outs
        rm(file)
    end
end
