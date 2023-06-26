abstract type Recorder end

mutable struct CSVRecorder <: Recorder
    filename::String
    primal_variables::AbstractArray{Symbol}
    dual_variables::AbstractArray{Symbol}
    filterfn::Function
end

function record(recorder::CSVRecorder, model::JuMP.Model, id::Int64)
    if !isfile(recorder.filename)
        open(recorder.filename, "w") do f
            write(f, "id")
            for p in recorder.primal_variables
                write(f, ",$p")
            end
            for p in recorder.dual_variables
                write(f, ",dual_$p")
            end
            write(f, "\n")
        end
    end
    open(recorder.filename, "a") do f
        write(f, "$id")
        for p in recorder.primal_variables
            val = MOI.get(model, MOI.VariablePrimal(), model[p])
            write(f, ",$val")
        end
        for p in recorder.dual_variables
            val = MOI.get(model, MOI.ConstraintDual(), model[p])
            write(f, ",$val")
        end
        write(f, "\n")
    end
end

function CSVRecorder(filename::String; primal_variables=[], dual_variables=[], filterfn=(model)-> termination_status(model) == MOI.OPTIMAL)
    return CSVRecorder(filename, primal_variables, dual_variables, filterfn)
end

function update_model!(model::JuMP.Model, p::VariableRef, val::T) where {T<:Real}
    MOI.set(model, POI.ParameterValue(), p, val)
end

function update_model!(model::JuMP.Model, p::VariableRef, val::AbstractArray{T}) where {T<:Real}
    MOI.set(model, POI.ParameterValue(), p, val)
end

function update_model!(model::JuMP.Model, pairs::Union{Base.Iterators.Zip, Base.Iterators.Pairs})
    for (p, val) in pairs
        update_model!(model, p, val)
    end
end

function solve_and_record(model::JuMP.Model, pairs::Union{Base.Iterators.Zip, Base.Iterators.Pairs}, recorder::Recorder, id::Int64)
    update_model!(model, pairs)
    optimize!(model)
    if recorder.filterfn(model)
        record(recorder, model, id)
    end
    return nothing
end

function solve_batch(model::JuMP.Model, problem_iterator::Union{Base.Iterators.Zip, Base.Iterators.Pairs}, recorder::Recorder)
    for (id, pairs) in problem_iterator
        solve_and_record(model, pairs, recorder, id)
    end
    return nothing
end
