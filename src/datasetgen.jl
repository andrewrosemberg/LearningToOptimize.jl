abstract type FileType end

mutable struct RecorderFile{T<:FileType}
    filename::String
end

filename(recorder_file::RecorderFile) = recorder_file.filename

termination_status_filter(status) = status == MOI.OPTIMAL || status == MOI.SLOW_PROGRESS || status == MOI.LOCALLY_SOLVED || status == MOI.ITERATION_LIMIT
primal_status_filter(status) = status == MOI.FEASIBLE_POINT
dual_status_filter(status) = status == MOI.FEASIBLE_POINT

filter_fn(model) = termination_status_filter(termination_status(model)) && primal_status_filter(primal_status(model)) && dual_status_filter(dual_status(model))

"""
    Recorder(filename; primal_variables=[], dual_variables=[], filterfn=(model)-> termination_status(model) == MOI.OPTIMAL)

Recorder of optimization problem solutions.
"""
mutable struct Recorder{T<:FileType}
    recorder_file::RecorderFile{T}
    recorder_file_input::RecorderFile{T}
    primal_variables::Vector
    dual_variables::Vector
    filterfn::Function

    function Recorder{T}(
        filename::String;
        filename_input::String=filename * "_input_",
        primal_variables=[],
        dual_variables=[],
        filterfn=filter_fn,
    ) where {T<:FileType}
        return new{T}(RecorderFile{T}(filename), RecorderFile{T}(filename_input), primal_variables, dual_variables, filterfn)
    end
end

filename(recorder::Recorder) = filename(recorder.recorder_file)
filename_input(recorder::Recorder) = filename(recorder.recorder_file_input)
get_primal_variables(recorder::Recorder) = recorder.primal_variables
get_dual_variables(recorder::Recorder) = recorder.dual_variables
get_filterfn(recorder::Recorder) = recorder.filterfn

function similar(recorder::Recorder{T}) where {T<:FileType}
    return Recorder{T}(
        filename(recorder); 
        filename_input=filename_input(recorder),
        primal_variables=get_primal_variables(recorder),
        dual_variables=get_dual_variables(recorder),
        filterfn=get_filterfn(recorder),
    )
end

function set_primal_variable!(recorder::Recorder, p::Vector)
    recorder.primal_variables = p
end

function set_dual_variable!(recorder::Recorder, p::Vector)
    recorder.dual_variables = p
end

abstract type AbstractProblemIterator end

"""
    ProblemIterator(ids::Vector{UUID}, pairs::Dict{VariableRef, Vector{Real}})

Iterator for optimization problem instances.
"""
struct ProblemIterator{T<:Real} <: AbstractProblemIterator
    model::JuMP.Model
    ids::Vector{UUID}
    pairs::Dict{VariableRef,Vector{T}}
    early_stop::Function
    function ProblemIterator(
        ids::Vector{UUID}, pairs::Dict{VariableRef,Vector{T}}, early_stop::Function=(args...) -> false
    ) where {T<:Real}
        model = JuMP.owner_model(first(keys(pairs)))
        for (p, val) in pairs
            @assert length(ids) == length(val)
        end
        return new{T}(model, ids, pairs, early_stop)
    end
end

function ProblemIterator(pairs::Dict{VariableRef,Vector{T}}; early_stop::Function=(args...) -> false) where {T<:Real}
    ids = [uuid1() for _ in 1:length(first(values(pairs)))]
    return ProblemIterator(ids, pairs, early_stop)
end

"""
    save(problem_iterator::ProblemIterator, filename::String, file_type::Type{T})

Save optimization problem instances to a file.
"""
function save(
    problem_iterator::AbstractProblemIterator, filename::String, file_type::Type{T}
) where {T<:FileType}
    save(
        (;
            id=problem_iterator.ids,
            zip(
                Symbol.(name.(keys(problem_iterator.pairs))), values(problem_iterator.pairs)
            )...,
        ),
        filename,
        file_type,
    )
    return nothing
end

"""
    update_model!(model::JuMP.Model, p::VariableRef, val::Real)

Update the value of a parameter in a JuMP model.
"""
function update_model!(model::JuMP.Model, p::VariableRef, val)
    return MOI.set(model, POI.ParameterValue(), p, val)
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
    solve_and_record(problem_iterator::ProblemIterator, recorder::Recorder, idx::Integer)

Solve an optimization problem and record the solution.
"""
function solve_and_record(
    problem_iterator::ProblemIterator, recorder::Recorder, idx::Integer
)
    model = problem_iterator.model
    update_model!(model, problem_iterator.pairs, idx)
    optimize!(model)
    status = recorder.filterfn(model)
    early_stop_bool = problem_iterator.early_stop(model, status, recorder)
    if status
        record(recorder, problem_iterator.ids[idx])
        return 1, early_stop_bool
    end
    return 0, early_stop_bool
end

"""
    solve_batch(problem_iterator::AbstractProblemIterator, recorder)

Solve a batch of optimization problems and record the solutions.
"""
function solve_batch(
    problem_iterator::AbstractProblemIterator, recorder
)
    successfull_solves = 0.0
    for idx in 1:length(problem_iterator.ids)
        _success_bool, early_stop_bool = solve_and_record(problem_iterator, recorder, idx)
        if _success_bool == 1
            successfull_solves += 1
        end
        if early_stop_bool
            break
        end
    end
    successfull_solves = successfull_solves / length(problem_iterator.ids)

    @info "Recorded $(successfull_solves * 100) % of $(length(problem_iterator.ids)) problems"
    return successfull_solves
end
