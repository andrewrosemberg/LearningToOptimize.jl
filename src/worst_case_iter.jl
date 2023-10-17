# Nonconvex needs a minimization objective function that only receives the decision vector.
function primal_objective(parameter_values, parameters, filter_fn; penalty=1e8)
    model = owner_model(first(parameters))
    for (i, p) in enumerate(parameters)
        update_model!(model, p, parameter_values[i])
    end

    # Solve the model
    status = false
    try 
        JuMP.optimize!(model)

        status = filter_fn(model)
    catch e
        @warn "Error in solve_and_record: $e"
    end

    objective = if status
        JuMP.objective_value(model)
    else
        - penalty
    end

    return objective
end

function save_input(parameters, _recorder, id)
    recorder = similar(_recorder)
    set_primal_variable!(recorder, parameters)
    set_dual_variable!(recorder, [])
    record(recorder, id; input=true)
    return nothing
end

mutable struct StorageCallbackObjective <: Function
    success_solves::Int
    fcalls::Int
    parameters::Vector{VariableRef}
    filter_fn::Function
    recorder::Recorder
end
StorageCallbackObjective(parameters, filter_fn, recorder) = StorageCallbackObjective(0, 0,
    parameters,
    filter_fn,
    recorder
)

function (callback::StorageCallbackObjective)(parameter_values)
    Zygote.@ignore callback.fcalls += 1

    obj = primal_objective(parameter_values, callback.parameters, callback.filter_fn)

    Zygote.@ignore begin
        if obj > 0
            callback.success_solves += 1
            id = uuid1(); record(callback.recorder, id); save_input(callback.parameters, callback.recorder, id)
        end
    end
    Zygote.@ignore @info "Iter: $(callback.fcalls):" obj
    return - obj
end

function solve_and_record(
    problem_iterator::WorstCaseProblemIterator{T}, recorder::Recorder, idx::Integer; maxiter=1000,
) where {T<:NonconvexCore.AbstractOptimizer}
    # Build Primal
    model, parameters = problem_iterator.primal_builder!(;recorder=recorder)
    (min_demands, max_demands, max_total_volume, starting_point) = problem_iterator.set_iterator!(idx)

    storage_objective_function = StorageCallbackObjective(
        parameters, recorder.filterfn,
        recorder
    )

    if haskey(problem_iterator.ext, :best_solution)
        starting_point = problem_iterator.ext[:best_solution]
    end

    # Build Nonconvex optimization model:
    model_non = Nonconvex.Model()
    set_objective!(model_non, storage_objective_function, flags = [:expensive])
    addvar!(model_non, min_demands, max_demands)
    # add_ineq_constraint!(model_non, x -> sum(x .^ 2) - max_total_volume ^ 2)

    # Optimize model_non:
    r_Nonconvex = if !isnothing(problem_iterator.options)
        optimize(model_non, problem_iterator.optimizer, starting_point; options = problem_iterator.options)
    else
        optimize(model_non, problem_iterator.optimizer, starting_point)
    end

    problem_iterator.ext[:best_solution] = r_Nonconvex.minimizer
    # best_profit = -r_Nonconvex.minimum

    return storage_objective_function.success_solves / storage_objective_function.fcalls, false
end