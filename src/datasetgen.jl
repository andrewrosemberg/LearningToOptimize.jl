function update_model!(model::Model, p::Parameter, val::T) where {T<:Real}
    MOI.set(model, ParameterValue(), p, val)
end

function update_model!(model::Model, p::Parameter, val::AbstractArray{T}) where {T<:Real}
    MOI.set(model, ParameterValue(), p, val)
end

function update_model!(model::Model, pairs::Union{Base.Iterators.Zip, Base.Iterators.Pairs})
    for (p, val) in pairs
        update_model!(model, p, val)
    end
end

function solve_and_record(model::Model, pairs::Union{Base.Iterators.Zip, Base.Iterators.Pairs}, recorder::Function, filterfn::Function)
    update_model!(model, pairs)
    optimize!(model)
    if filterfn(model)
        recorder(model, pairs)
    end
    return nothing
end
