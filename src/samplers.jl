"""
    function line_sampler(
        original_parameters::Vector{T},
        parameter_indexes::Vector{F},
        range_p::AbstractVector{T},
    ) where {T<:Real,F<:Integer}

This sampler returns a set of parameters that for a line in one dimension of the parameter space. 
The idea is to change the value of one parameter and keep the rest constant.
"""
function line_sampler(
    original_parameters::Vector{T},
    parameter_indexes::AbstractVector{F},
    range_p::AbstractVector{T},
) where {T<:Real,F<:Integer}
    parameters = hcat(fill(original_parameters, length(range_p))...)

    for parameter_index in parameter_indexes
        parameters[parameter_index, :] = [original_parameters[parameter_index] * mul for mul in range_p]
    end

    return parameters
end

function line_sampler(
    original_parameters::Vector{T},
    range_p::AbstractVector{T},
) where {T<:Real}
    parameters = zeros(T, length(original_parameters), length(range_p) * (1 + length(original_parameters)))
    parameters[:, 1:length(range_p)] = line_sampler(original_parameters, 1:length(original_parameters), range_p)

    for parameter_index=1:length(original_parameters)
        parameters[:, length(range_p) * parameter_index + 1:length(range_p) * (parameter_index + 1)] = line_sampler(original_parameters, [parameter_index], range_p)
    end

    return parameters
end

"""
    function box_sampler(
        original_parameter::T,
        num_p::F,
        range_p::AbstractVector{T}=0.8:0.01:1.25,
    ) where {T<:Real,F<:Integer}
    
Uniformly sample values around the original parameter value over a discrete range inside a box.
"""
function box_sampler(
    original_parameter::T,
    num_p::F,
    range_p::AbstractVector{T}=0.8:0.01:1.25,
) where {T<:Real,F<:Integer}
    # parameter sampling
    parameter_samples =
        original_parameter * rand(range_p, num_p)
    return parameter_samples
end

function box_sampler(
    original_parameters::Vector{T},
    num_p::F,
    range_p::AbstractVector{T}=0.8:0.01:1.25,
) where {T<:Real,F<:Integer}
    # parameter sampling
    return vcat([box_sampler(p, num_p, range_p)' for p in original_parameters]...)
end

function general_sampler(
    original_parameters::Vector{T},
    line_sampler_range::AbstractVector{T}=1.01:0.01:1.25,
    box_sampler_num_p::Union{F, Nothing}=nothing,
    box_sampler_range::AbstractVector{T}=0.7:0.01:1.25,
) where {T<:Real,F<:Integer}
    if box_sampler_num_p == nothing
        return line_sampler(original_parameters, line_sampler_range)
    else
        return hcat(line_sampler(original_parameters, line_sampler_range), box_sampler(original_parameters, box_sampler_num_p, box_sampler_range))
    end
end