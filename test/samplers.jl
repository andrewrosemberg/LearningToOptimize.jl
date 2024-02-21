function test_line_sampler(; num_p=10, range_p = 1:0.01:1.1)
    original_parameter = rand(num_p)
    for parameter_index = 1:num_p
        parameters = L2O.line_sampler(
            original_parameter,
            [parameter_index],
            range_p,
        )
        @test parameters[parameter_index, 1] == original_parameter[parameter_index]
        @test parameters[parameter_index, :] == [original_parameter[parameter_index] * mul for mul in range_p]
    end
    parameters = L2O.line_sampler(
        original_parameter,
        range_p,
    )
    @test size(parameters) == (10, length(range_p) * (1 + num_p))
    return nothing
end

function test_box_sampler(; num_p=10, num_s=5, max_multiplier=3.0, min_multiplier=0.0, step_multiplier=0.1)
    original_parameter = rand(num_p)
    parameters = box_sampler(original_parameter, num_s, min_multiplier:step_multiplier:max_multiplier)
    @test size(parameters) == (num_p, num_s)
    @test all(parameters .>= original_parameter * min_multiplier)
    @test all(parameters .<= original_parameter * max_multiplier)
    return nothing
end

function test_general_sampler(; num_p=10, num_s=5, range_p=1.01:0.01:1.25)
    original_parameter = rand(num_p)
    parameters = general_sampler(
        original_parameter;
        samplers=[
            (original_parameters) -> scaled_distribution_sampler(original_parameters, num_s),
            line_sampler, 
            (original_parameters) -> box_sampler(original_parameters, num_s),
        ]
    )
    @test size(parameters) == (num_p, 2 * num_s + length(range_p) * (1 + num_p))
    return nothing
end

function test_load_parameters_model(;num_p=10, num_v=5)
    model = JuMP.Model()
    @variable(model, 0 <= x[1:num_v] <= 1)
    @variable(model, p[1:num_p] in MOI.Parameter.(1.0))
    @constraint(model, cons, sum(x) + sum(p) >= 3)
    @objective(model, Min, 2x)

    parameters, vals = L2O.load_parameters(model)
    @test length(parameters) == num_p
    @test length(vals) == num_p
    @test all(vals .== 1.0)
    @test all(parameters .== p)
    return nothing
end

function test_load_parameters()
    file="pglib_opf_case5_pjm_DCPPowerModel_POI_load.mof.json"
    parameters, vals = L2O.load_parameters(file)
    @test length(parameters) == 3
    @test length(vals) == 3
    @test all(vals .== 1.0)
    return nothing
end

function test_general_sampler_file(file::AbstractString="pglib_opf_case5_pjm_DCPPowerModel_POI_load.mof.json"; 
    num_s=5, range_p=1.01:0.01:1.25, 
    cache_dir=mktempdir(),
    batch_id=uuid1(),
    save_file = joinpath(cache_dir, split(split(file, ".mof.json")[1], "/")[end] * "_input_" * string(batch_id)),
    filetype=CSVFile
)
    _, vals = L2O.load_parameters(file)
    num_p=length(vals)
    problem_iterator = general_sampler(
        file;
        samplers=[
            (original_parameters) -> scaled_distribution_sampler(original_parameters, num_s),
            line_sampler, 
            (original_parameters) -> box_sampler(original_parameters, num_s),
        ],
        save_file=save_file,
        batch_id=batch_id,
        filetype=filetype
    )
    @test length(problem_iterator.ids) == 2 * num_s + length(range_p) * (1 + num_p)
    @test length(problem_iterator.pairs) == num_p

    input_table = load(save_file, filetype)
    @test size(input_table) == (length(problem_iterator.ids), num_p + 1)

    return save_file
end
