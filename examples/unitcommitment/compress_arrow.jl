using Arrow
using DataFrames

# Data Parameters
case_name = "case300"
path_dataset = joinpath(pwd(), "examples", "unitcommitment", "data")
case_file_path = path_dataset # joinpath(path_dataset, case_name)

# Load input and output data tables
iter_files = readdir(joinpath(case_file_path))
file_ins = [
    joinpath(case_file_path, file) for file in iter_files if occursin("input", file)
]

batch_ids = [split(split(file, "_")[end], ".")[1] for file in file_ins]

file_name = split(split(file_ins[1], "_input")[1], "/")[end]

# compress output files per batch id

for batch_id in batch_ids

    file_outs = [
        joinpath(case_file_path, file)
        for file in iter_files
        if occursin("output", file) && occursin(batch_id, file)
    ]
    if length(file_outs) == 1
        continue
    end

    # Load input and output data tables
    output_table = Arrow.Table(file_outs)

    # Save compressed files
    Arrow.write(
        joinpath(case_file_path, "$(file_name)_output_" * batch_id * ".arrow"),
        output_table,
    )

    # Delete uncompressed files
    for file in file_outs
        rm(file)
    end
end
