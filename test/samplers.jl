function test_line_sampler(; num_p=10, range_p = 1:0.01:1.1)
    original_parameter = rand(10)
    for parameter_index = 1:num_p
        parameters = line_sampler(
            original_parameter,
            [parameter_index],
            range_p,
        )
        @test parameters[parameter_index, 1] == original_parameter[parameter_index]
        @test parameters[parameter_index, :] == [original_parameter[parameter_index] * mul for mul in range_p]
    end
end

function test_box_sampler(; num_p=10, max_multiplier=3.0, min_multiplier=0.0, step_multiplier=0.1)
    original_parameter = rand(10)
    parameters = box_sampler(original_parameter, num_p, min_multiplier:step_multiplier:max_multiplier)
    @test size(parameters) == (10, num_p)
    @test all(parameters .>= original_parameter * min_multiplier)
    @test all(parameters .<= original_parameter * max_multiplier)

    return nothing
end