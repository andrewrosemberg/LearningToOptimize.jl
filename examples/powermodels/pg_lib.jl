using Downloads
using PowerModels
using JuMP, HiGHS
import ParametricOptInterface as POI

"""
    createvarrefs!(sp::JuMP.Model, pm::AbstractPowerModel)

create ref for anonimous variables on model
"""
function createvarrefs!(sp::JuMP.Model, pm::AbstractPowerModel)
    for listvarref in values(PowerModels.var(pm))
        for variableref in values(listvarref)
            if typeof(variableref) == JuMP.VariableRef
                sp[Symbol(name(variableref))] = variableref
            end
        end
    end
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
    original_load = network_data["load"]["1"]["pd"]
    network_data["load"]["1"]["pd"] = p = @variable(model, _p in POI.Parameter(1.0))

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
        collect(1:num_p), Dict(p => collect(1.0:num_p) .* original_load)
    )

    # Create ref for anonimous variables on model
    createvarrefs!(model, pm)

    # Solve the problem and return the number of successfull solves
    file = joinpath(data_dir, "test.$(string(filetype))")
    number_generators = length(network_data["gen"])
    recorder = Recorder{filetype}(
        file; primal_variables=[Symbol("0_pg[$i]") for i in 1:number_generators]
    )
    return solve_batch(model, problem_iterator, recorder), number_generators
end

