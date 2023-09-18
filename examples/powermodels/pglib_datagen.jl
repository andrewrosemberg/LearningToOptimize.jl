using PGLib
using PowerModels
using JuMP, HiGHS
import ParametricOptInterface as POI

"""
    return_variablerefs(pm::AbstractPowerModel)

return all variablerefs on pm
"""
function return_variablerefs(pm::AbstractPowerModel)
    return vcat(
        [
            [
                variableref for
                variableref in values(listvarref) if typeof(variableref) == JuMP.VariableRef
            ] for listvarref in values(PowerModels.var(pm))
        ]...,
    )
end

"""
    load_sampler(original_load::T, num_p::Int, max_multiplier::T=3.0, min_multiplier::T=0.0, step_multiplier::T=0.1)

Load sampling
"""
function load_sampler(
    original_load::T,
    num_p::Int;
    max_multiplier::T=2.5,
    min_multiplier::T=0.0,
    step_multiplier::T=0.1,
) where {T<:Real}
    # Load sampling
    load_samples =
        original_load * rand(min_multiplier:step_multiplier:max_multiplier, num_p)
    return load_samples
end

function load_parameter_factory(model, indices; load_set=nothing)
    if isnothing(load_set)
        return @variable(
            model, _p[i in indices]
        )
    end
    return @variable(
        model, _p[i in indices] in load_set
    )
end

function pm_primal_builder!(model, parameters, network_data, network_formulation; recorder=nothing)
    num_loads = length(network_data["load"])
    for (i, l) in enumerate(values(network_data["load"]))
        l["pd"] = parameters[i]
        l["qd"] = parameters[num_loads+i]
    end

    # Instantiate the model
    pm = instantiate_model(
        network_data,
        network_formulation,
        PowerModels.build_opf;
        setting=Dict("output" => Dict("duals" => true)),
        jump_model=model,
    )

    if !isnothing(recorder)
        variable_refs = return_variablerefs(pm)
        for variableref in variable_refs
            set_name(variableref, replace(name(variableref), "," => "_"))
        end
        set_primal_variable!(recorder, variable_refs)
        # set_dual_variable!(recorder, [cons for cons in values(pm["constraint"])])
    end

    return nothing
end

function load_set_iterator!(model, parameters, idx, original_load)
    for (i, p) in enumerate(parameters)
        @constraint(model, p <= original_load[i] * (1.0 + 0.01 * idx))
        @constraint(model, p >= 0.0)
    end
end

"""
    generate_dataset_pglib(data_dir::AbstractString, case_name::AbstractString; download_files::Bool=true, filetype::Type{FileType},
    num_p::Int=10
)

Generate dataset for pglib case_name with num_p problems and save it in data_dir
"""
function generate_dataset_pglib(
    data_dir,
    case_name;
    filetype=CSVFile,
    num_p=10,
    load_sampler=load_sampler,
    network_formulation=DCPPowerModel,
    solver = () -> POI.Optimizer(HiGHS.Optimizer()),
)
    # Download file
    matpower_case_name = case_name * ".m"

    # save folder
    data_sim_dir = joinpath(data_dir, string(network_formulation))
    if !isdir(data_sim_dir)
        mkdir(data_sim_dir)
    end

    # Read data
    network_data = make_basic_network(pglib(matpower_case_name))

    # The problem to iterate over
    model = Model(solver)
    MOI.set(model, MOI.Silent(), true)

    # Save original load value and Link POI
    num_loads = length(network_data["load"])
    original_load = vcat(
        [l["pd"] for l in values(network_data["load"])],
        [l["qd"] for l in values(network_data["load"])],
    )
    p = load_parameter_factory(model, 1:(num_loads * 2); load_set=POI.Parameter.(original_load))

    # Define batch id
    batch_id = string(uuid1())
    @info "Batch ID: $batch_id"

    # Build model and Recorder
    file = joinpath(data_sim_dir, case_name * "_" * string(network_formulation) * "_output_" * batch_id * "." * string(filetype))
    recorder = Recorder{filetype}(file)
    pm_primal_builder!(model, p, network_data, network_formulation; recorder=recorder)

    # The problem iterator
    problem_iterator = ProblemIterator(
        Dict(
            p .=> load_sampler.(original_load, num_p),
        ),
    )

    save(
        problem_iterator,
        joinpath(data_sim_dir, case_name * "_" * string(network_formulation) * "_input_" * batch_id * "." * string(filetype)),
        filetype,
    )

    # Solve the problem and return the number of successfull solves
    return solve_batch(problem_iterator, recorder),
        length(recorder.primal_variables),
        length(original_load),
        batch_id
end

function test_pglib_datasetgen(path::AbstractString, case_name::AbstractString, num_p::Int)
    @testset "Dataset Generation pglib case" begin
        network_formulation = DCPPowerModel
        success_solves, number_variables, number_parameters, batch_id = generate_dataset_pglib(
            path, case_name; num_p=num_p, network_formulation=network_formulation
        )
        file_in = joinpath(path, string(network_formulation), case_name * "_" * string(network_formulation) * "_input_" * batch_id * ".csv")
        file_out = joinpath(path, string(network_formulation), case_name * "_" * string(network_formulation) * "_output_" * batch_id * ".csv")
        # Check if problem iterator was saved
        @test isfile(file_in)
        @test length(readdlm(file_in, ',')[:, 1]) == num_p + 1
        @test length(readdlm(file_in, ',')[1, :]) == 1 + number_parameters

        # Check if the number of successfull solves is equal to the number of problems saved
        @test isfile(file_out)
        @test length(readdlm(file_out, ',')[:, 1]) == num_p * success_solves + 1
        @test length(readdlm(file_out, ',')[1, :]) == number_variables + 2

        return file_in, file_out
    end
end
