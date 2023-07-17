using Downloads
using PowerModels
using JuMP, HiGHS
import ParametricOptInterface as POI

"""
    return_variablerefs(pm::AbstractPowerModel)

return all variablerefs on pm
"""
function return_variablerefs(pm::AbstractPowerModel)
    return vcat([[variableref for variableref in values(listvarref) if typeof(variableref) == JuMP.VariableRef] for listvarref in values(PowerModels.var(pm))]...)
end

"""
    generate_dataset_pglib(data_dir::AbstractString, case_name::AbstractString; download_files::Bool=true, filetype::Type{RecorderFile},
    num_p::Int=10
)

Generate dataset for pglib case_name with num_p problems and save it in data_dir
"""
function generate_dataset_pglib(
    data_dir,
    case_name;
    filetype=CSVFile,
    download_files=true,
    num_p=10,
)
    case_file_path = joinpath(data_dir, case_name)
    if download_files && !isfile(case_file_path)
        Downloads.download(
            "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/01681386d084d8bd03b429abcd1ee6966f68b9a3/" *
            case_name,
            case_file_path,
        )
    end

    # Read data
    network_data = PowerModels.parse_file(case_file_path)

    # The problem to iterate over
    model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))

    # Save original load value and Link POI
    original_load = [l["pd"] for l in values(network_data["load"])]
    p = @variable(model, _p[i=1:length(network_data["load"])] in POI.Parameter.(original_load)) # vector of parameters
    for (i,l) in enumerate(values(network_data["load"]))
        l["pd"] = p[i]
    end

    # Instantiate the model
    pm = instantiate_model(
        network_data,
        DCPPowerModel,
        PowerModels.build_opf;
        setting=Dict("output" => Dict("duals" => true)),
        jump_model=model,
    )

    # The problem iterator
    problem_iterator = ProblemIterator(
        collect(1:num_p), Dict(p .=> [collect(0.1:0.5:0.5 * num_p) * original_load[i] for i in 1:length(network_data["load"])])
    )

    # Solve the problem and return the number of successfull solves
    file = joinpath(data_dir, "test.$(string(filetype))")
    variable_refs = return_variablerefs(pm)
    for variableref in variable_refs
        set_name(variableref, replace(name(variableref), "," => "_"))
    end
    number_vars = length(variable_refs)
    recorder = Recorder{filetype}(
        file; primal_variables=variable_refs
    )
    return solve_batch(model, problem_iterator, recorder), number_vars
end

