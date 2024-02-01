using Arrow
using DataFrames

# Data Parameters
case_name = "case300"
date = "2017-01-01"
horizon = 2
case_file_path = joinpath(
    dirname(@__FILE__), "data", case_name, date, "h" * string(horizon)
)

# Load input and output data tables
iter_files = readdir(case_file_path)

iter_files = [
    file for file in iter_files if occursin(case_name, file) && occursin("arrow", file)
]

file_ins = [
    joinpath(case_file_path, file) for file in iter_files if occursin("input", file)
]

batch_ids = [split(split(file, "_")[end], ".")[1] for file in file_ins]

@info "Compressing files in $(case_file_path)" batch_ids

file_name = split(split(file_ins[1], "_input")[1], "/")[end]

# move input files to input folder
for file in iter_files
    if occursin("input", file)
        mv(joinpath(case_file_path, file), joinpath(case_file_path, "input", file), force=true)
    end
end

# compress output files per batch id
for batch_id in batch_ids
    file_outs_names = [
        file
        for file in iter_files
        if occursin("output", file) && occursin(batch_id, file)
    ]

    file_outs = [ joinpath(case_file_path, file) for file in file_outs_names ]
    
    if length(file_outs_names) == 1
        mv(file_outs[1], joinpath(case_file_path, "output", file_outs_names[1]), force=true)
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
