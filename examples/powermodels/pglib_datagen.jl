using PGLib
using PowerModels
using JuMP, HiGHS
import ParametricOptInterface as POI
using LinearAlgebra

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

function add_names_pm(pm::AbstractPowerModel)
    variable_refs = return_variablerefs(pm)
    for variableref in variable_refs
        set_name(variableref, replace(name(variableref), "," => "_"))
    end
end

"""
    load_sampler(original_load::T, num_p::Int, max_multiplier::T=3.0, min_multiplier::T=0.0, step_multiplier::T=0.1)

Load sampling
"""
function load_sampler(
    original_load::T,
    num_p::F,
    ::F,
    ::F;
    max_multiplier::T=2.5,
    min_multiplier::T=0.0,
    step_multiplier::T=0.1,
) where {T<:Real, F<:Integer}
    # Load sampling
    load_samples =
        original_load * rand(min_multiplier:step_multiplier:max_multiplier, num_p)
    return load_samples
end

"""
    line_sampler(original_parameter::T, num_p::Int, parameter_index::F, num_inputs::F, line_index::F; step_multiplier::T=0.1)

line_sampler is a function to help generate a dataset for varying parameter values. It has two modes:
 - If line_index is not outside the parameter index range: 
    Return an incremental vector for the parameter at parameter_index and an unchanged parameter for the rest;
 - If line_index is outside the parameter index range:
    Return an incremental vector for all parameters. 
"""
function line_sampler(
    original_parameter::T, 
    num_p::F, 
    parameter_index::F, 
    num_inputs::F, 
    line_index::F; 
    step_multiplier::T=1.01,
)  where {T<:Real, F<:Integer}
    # parameter sampling
    num_parameters = floor(Int, num_inputs / 2)
    if (parameter_index == line_index) || (parameter_index - num_parameters == line_index) || (line_index == num_inputs + 1)
        return [original_parameter * step_multiplier ^ (j) for j in 1:num_p]
    else
        return ones(num_p) * original_parameter
    end
end

"""
    load_parameter_factory(model, indices; load_set=nothing)

load_parameter_factory is a function to help generate a parameter vector for the problem iterator.
"""
function load_parameter_factory(model, indices; load_set=nothing)
    if isnothing(load_set)
        return @variable(
            model, _p[i=indices]
        )
    end
    return @variable(
        model, _p[i=indices] in load_set
    )
end

"""
    pm_primal_builder!(model, parameters, network_data, network_formulation; recorder=nothing)

pm_primal_builder! is a function to help build a PowerModels model and update recorder.
"""
function pm_primal_builder!(model, parameters, network_data, network_formulation; recorder=nothing)
    num_loads = length(network_data["load"])
    for (str_i, l) in network_data["load"]
        i = parse(Int, str_i)
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
        return model, parameters, variable_refs
    end

    return nothing
end

"""
    load_set_iterator!(model, parameters, idx, original_load)

load_set_iterator! is a function to help iterate over the load set. Used in the worst case generator.
"""
function load_set_iterator!(model, parameters, idx, original_load)
    for (i, p) in enumerate(parameters)
        @constraint(model, p <= original_load[i] * (1.0 + 0.1 * idx))
        @constraint(model, p >= original_load[i] * (1.0 - 0.1 * idx))
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
    internal_load_sampler=load_sampler,
    network_formulation=DCPPowerModel,
    optimizer = () -> POI.Optimizer(HiGHS.Optimizer()),
    filterfn=L2O.filter_fn,
    early_stop_fn = (model, status, recorder) -> false,
    batch_id = string(uuid1())
)
    @info "Batch ID: $batch_id"

    # save folder
    data_sim_dir = joinpath(data_dir, string(network_formulation))
    mkpath(data_sim_dir)

    # Read data
    matpower_case_name = case_name * ".m"
    network_data = make_basic_network(pglib(matpower_case_name))

    # The problem to iterate over
    model = JuMP.Model(optimizer)
    MOI.set(model, MOI.Silent(), true)

    # Save original load value and Link POI
    num_loads = length(network_data["load"])
    num_inputs = num_loads * 2
    original_load = vcat(
        [network_data["load"]["$l"]["pd"] for l=1:num_loads],
        [network_data["load"]["$l"]["qd"] for l=1:num_loads],
    )
    p = load_parameter_factory(model, 1:num_inputs; load_set=POI.Parameter.(original_load))

    # Build model and Recorder
    file = joinpath(data_sim_dir, case_name * "_" * string(network_formulation) * "_output_" * batch_id * "." * string(filetype))
    recorder = Recorder{filetype}(file; filterfn=filterfn)
    pm_primal_builder!(model, p, network_data, network_formulation; recorder=recorder)

    # The problem iterator
    pairs = Dict{VariableRef, Vector{Float64}}()
    for i in 1:num_inputs
        pairs[p[i]] = internal_load_sampler(original_load[i], num_p, i, num_inputs)
    end
    problem_iterator = ProblemIterator(
        pairs;
        early_stop=early_stop_fn
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

function generate_worst_case_dataset_Nonconvex(data_dir,
    case_name;
    filetype=CSVFile,
    num_p=10,
    network_formulation=DCPPowerModel,
    optimizer = () -> POI.Optimizer(HiGHS.Optimizer()),
    algorithm = NLoptAlg(:LN_BOBYQA),
    options = NLoptOptions(maxeval=10),
)
    # save folder
    data_sim_dir = joinpath(data_dir, string(network_formulation))
    if !isdir(data_sim_dir)
        mkdir(data_sim_dir)
    end

    # Read data
    matpower_case_name = case_name * ".m"
    network_data = make_basic_network(pglib(matpower_case_name))

    # The problem to iterate over
    model = JuMP.Model(() -> optimizer())
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

    # File names
    file_input = joinpath(data_sim_dir, case_name * "_" * string(network_formulation) * "_input_" * batch_id * "." * string(filetype))
    file_output = joinpath(data_sim_dir, case_name * "_" * string(network_formulation) * "_output_" * batch_id * "." * string(filetype))
    recorder = Recorder{filetype}(
        file_output; filename_input=file_input,
        primal_variables=[], dual_variables=[]
    )

    # Build model
    model, parameters, variable_refs = pm_primal_builder!(model, p, network_data, network_formulation; recorder=recorder)
    function _primal_builder!(;recorder=nothing)
        if !isnothing(recorder)
            set_primal_variable!(recorder, variable_refs)
        end

       return model, parameters
    end

    # Set iterator
    function _set_iterator!(idx)
        _min_demands = original_load .- ones(num_loads * 2) .* 0.1 * idx
        _max_demands = original_load .+ ones(num_loads * 2) .* 0.1 * idx
        min_demands = min.(_min_demands, _max_demands)
        max_demands = max.(_min_demands, _max_demands)
        max_total_volume = (norm(max_demands, 2) + norm(min_demands, 2)) ^ 2
        starting_point = original_load
        return min_demands, max_demands, max_total_volume, starting_point
    end

    # The problem iterator
    problem_iterator = WorstCaseProblemIterator(
        [uuid1() for _ in 1:num_p], # will be ignored
        () -> nothing, # will be ignored
        _primal_builder!,
        _set_iterator!,
        algorithm;
        options = options
    )

    # Solve all problems and record solutions
    return solve_batch(problem_iterator, recorder),
        length(recorder.primal_variables),
        length(original_load),
        batch_id
end

function default_optimizer_factory()
    return () -> Ipopt.Optimizer()
end

function generate_worst_case_dataset(data_dir,
    case_name;
    filetype=CSVFile,
    num_p=10,
    network_formulation=DCPPowerModel,
    optimizer_factory = default_optimizer_factory,
    hook = nothing
)
    # save folder
    data_sim_dir = joinpath(data_dir, string(network_formulation))
    if !isdir(data_sim_dir)
        mkdir(data_sim_dir)
    end

    # Read data
    matpower_case_name = case_name * ".m"
    network_data = make_basic_network(pglib(matpower_case_name))

    # Parameter factory
    num_loads = length(network_data["load"])
    original_load = vcat(
        [l["pd"] for l in values(network_data["load"])],
        [l["qd"] for l in values(network_data["load"])],
    )
    parameter_factory = (model) -> load_parameter_factory(model, 1:(num_loads * 2))

    # Define batch id
    batch_id = string(uuid1())
    @info "Batch ID: $batch_id"

    # Build model
    primal_builder! = (model, parameters; recorder=nothing) -> pm_primal_builder!(model, parameters, network_data, network_formulation; recorder=recorder)

    # Set iterator
    set_iterator! = (model, parameters, idx) -> load_set_iterator!(model, parameters, idx, original_load)

    # The problem iterator
    problem_iterator = WorstCaseProblemIterator(
        [uuid1() for _ in 1:num_p],
        parameter_factory,
        primal_builder!,
        set_iterator!,
        optimizer_factory;
        hook=hook
    )

    # File names
    file_input = joinpath(data_sim_dir, case_name * "_" * string(network_formulation) * "_input_" * batch_id * "." * string(filetype))
    file_output = joinpath(data_sim_dir, case_name * "_" * string(network_formulation) * "_output_" * batch_id * "." * string(filetype))
    recorder = Recorder{filetype}(
        file_output; filename_input=file_input,
        primal_variables=[], dual_variables=[]
    )

    # Solve all problems and record solutions
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

function test_generate_worst_case_dataset(path::AbstractString, case_name::AbstractString, num_p::Int)
    @testset "Worst Case Dataset Generation pglib case" begin
        network_formulation = DCPPowerModel
        # Improve dataset
        success_solves, number_variables, number_parameters, batch_id = generate_worst_case_dataset(
            path, case_name; num_p=num_p, network_formulation=network_formulation
        )

        file_in = joinpath(path, string(network_formulation), case_name * "_" * string(network_formulation) * "_input_" * batch_id * ".csv")
        file_out = joinpath(path, string(network_formulation), case_name * "_" * string(network_formulation) * "_output_" * batch_id * ".csv")

        # Check if problem iterator was saved
        @test isfile(file_in)
        @test length(readdlm(file_in, ',')[:, 1]) >=  num_p * success_solves + 1
        @test length(readdlm(file_in, ',')[1, :]) == 1 + number_parameters

        # Check if the number of successfull solves is equal to the number of problems saved
        @test isfile(file_out)
        @test length(readdlm(file_out, ',')[:, 1]) >= num_p * success_solves + 1
        @test length(readdlm(file_out, ',')[1, :]) == number_variables + 2

        return file_in, file_out
    end
end

function test_generate_worst_case_dataset_Nonconvex(path::AbstractString, case_name::AbstractString, num_p::Int)
    @testset "WC Nonconvex Dataset Generation pglib case" begin
        network_formulation = DCPPowerModel
        # Improve dataset
        success_solves, number_variables, number_parameters, batch_id = generate_worst_case_dataset_Nonconvex(
            path, case_name; num_p=num_p, network_formulation=network_formulation
        )

        file_in = joinpath(path, string(network_formulation), case_name * "_" * string(network_formulation) * "_input_" * batch_id * ".csv")
        file_out = joinpath(path, string(network_formulation), case_name * "_" * string(network_formulation) * "_output_" * batch_id * ".csv")

        # Check if problem iterator was saved
        @test isfile(file_in)
        @test length(readdlm(file_in, ',')[:, 1]) >=  num_p * success_solves + 1
        @test length(readdlm(file_in, ',')[1, :]) == 1 + number_parameters

        # Check if the number of successfull solves is equal to the number of problems saved
        @test isfile(file_out)
        @test length(readdlm(file_out, ',')[:, 1]) >= num_p * success_solves + 1
        @test length(readdlm(file_out, ',')[1, :]) == number_variables + 2

        return file_in, file_out
    end
end
