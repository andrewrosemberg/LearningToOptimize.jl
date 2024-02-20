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
    parameter_indexes::Vector{F},
    range_p::AbstractVector{T},
) where {T<:Real,F<:Integer}
    parameters = hcat(fill(original_parameters, length(range_p))...)

    for parameter_index in parameter_indexes
        parameters[parameter_index, :] = [original_parameters[parameter_index] * mul for mul in range_p]
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
