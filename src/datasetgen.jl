"""
    Recorder

Abstract type for recorders of optimization problem solutions.
"""
abstract type Recorder end

"""
    CSVRecorder(filename; primal_variables=[], dual_variables=[], filterfn=(model)-> termination_status(model) == MOI.OPTIMAL)

Recorder type of optimization problem solutions to a CSV file.
"""
mutable struct CSVRecorder <: Recorder
    filename::String
    primal_variables::AbstractArray{Symbol}
    dual_variables::AbstractArray{Symbol}
    filterfn::Function

    function CSVRecorder(filename::String; primal_variables=[], dual_variables=[], filterfn=(model)-> termination_status(model) == MOI.OPTIMAL)
        return new(filename, primal_variables, dual_variables, filterfn)
    end
end

"""
    ProblemIterator(ids::Vector{Integer}, pairs::Dict{VariableRef, Vector{Real}})

Iterator for optimization problem instances.
"""
struct ProblemIterator{T<:Real, Z<:Integer}
    ids::Vector{Z}
    pairs::Dict{VariableRef, Vector{T}}
    function ProblemIterator(ids::Vector{Z}, pairs::Dict{VariableRef, Vector{T}}) where {T<:Real, Z<:Integer}
        for (p, val) in pairs
            @assert length(ids) == length(val)
        end
        return new{T, Z}(ids, pairs)
    end
end

"""
    record(recorder::CSVRecorder, model::JuMP.Model, id::Int64)

Record optimization problem solution to a CSV file.
"""
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

"""
    update_model!(model::JuMP.Model, p::VariableRef, val::Real)

Update the value of a parameter in a JuMP model.
"""
function update_model!(model::JuMP.Model, p::VariableRef, val::T) where {T<:Real}
    MOI.set(model, POI.ParameterValue(), p, val)
end

"""
    update_model!(model::JuMP.Model, p::VariableRef, val::AbstractArray{Real})

Update the value of a parameter in a JuMP model.
"""
function update_model!(model::JuMP.Model, p::VariableRef, val::AbstractArray{T}) where {T<:Real}
    MOI.set(model, POI.ParameterValue(), p, val)
end

"""
    update_model!(model::JuMP.Model, pairs::Dict, idx::Integer)

Update the values of parameters in a JuMP model.
"""
function update_model!(model::JuMP.Model, pairs::Dict, idx::Integer)
    for (p, val) in pairs
        update_model!(model, p, val[idx])
    end
end

"""
    solve_and_record(model::JuMP.Model, problem_iterator::ProblemIterator, recorder::Recorder, idx::Integer)

Solve an optimization problem and record the solution.
"""
function solve_and_record(model::JuMP.Model, problem_iterator::ProblemIterator, recorder::Recorder, idx::Integer)
    update_model!(model, problem_iterator.pairs, idx)
    optimize!(model)
    if recorder.filterfn(model)
        record(recorder, model, problem_iterator.ids[idx])
    end
    return nothing
end

"""
    solve_batch(model::JuMP.Model, problem_iterator::ProblemIterator, recorder::Recorder)

Solve a batch of optimization problems and record the solutions.
"""
function solve_batch(model::JuMP.Model, problem_iterator::ProblemIterator, recorder::Recorder)
    for idx in 1:length(problem_iterator.ids)
        solve_and_record(model, problem_iterator, recorder, idx)
    end
    return nothing
end
