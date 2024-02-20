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