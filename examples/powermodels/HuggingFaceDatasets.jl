using PyCall
using Conda
# Conda.add("huggingface_hub")

huggingface_hub = pyimport("huggingface_hub")

huggingface_hub.login(token=ENV["HUGGINGFACE_TOKEN"])

function download_dataset(organization, dataset, case_name, io_type; formulation="", cache_dir="./data/")
    dataset_url = joinpath(organization, dataset)
    if io_type ∉ ["input", "output"]
        throw(ArgumentError("io_type must be 'input' or 'output'."))
    end

    if io_type == "input"
        data_path = joinpath(case_name, "input")
    else
        if formulation == ""
            throw(ArgumentError("Formulation must be specified for 'output' data."))
        end
        data_path = joinpath(case_name, "output", formulation)
    end

    # Fetch the dataset from the provided URL
    huggingface_hub.snapshot_download(dataset_url, allow_patterns=["$data_path/*.arrow"], local_dir=cache_dir, repo_type="dataset", local_dir_use_symlinks=false)
    
    return nothing
end

cache_dir="./examples/powermodels/data/"
organization = "L2O"
dataset = "pglib_opf_solves"
case_name = "pglib_opf_case300_ieee"
formulation = "DCPPowerModel" # ACPPowerModel SOCWRConicPowerModel
io_type = "input"
download_dataset(organization, dataset, case_name, io_type; cache_dir=cache_dir)

io_type = "output"
download_dataset(organization, dataset, case_name, io_type; formulation=formulation , cache_dir=cache_dir)