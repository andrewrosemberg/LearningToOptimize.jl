struct WorstCaseProblemIterator <: AbstractProblemIterator
    ids::Vector{UUID}
    parameters::Function
    primal_builder!::Function
    set_iterator!::Function
    optimizer
    function WorstCaseProblemIterator(
        ids::Vector{UUID},
        parameters::Function,
        primal_builder!::Function,
        set_iterator!::Function,
        optimizer,
    )
        return new(ids, parameters, primal_builder!, set_iterator!, optimizer)
    end
end

function solve_and_record(
    problem_iterator::WorstCaseProblemIterator, recorder::Recorder, idx::Integer
)
    # Build Primal
    model = JuMP.Model()
    parameters = problem_iterator.parameters(model)
    problem_iterator.primal_builder!(model, parameters)
    
    # Parameter indices
    load_moi_idx = MOI.VariableIndex[i.index for i in parameters]

    # Dualize the model
    dual_st = Dualization.dualize(JuMP.backend(model), 
        variable_parameters = load_moi_idx
    )

    dual_model = dual_st.dual_model
    primal_dual_map = dual_st.primal_dual_map

    # Build Dual in JuMP
    jump_dual_model = JuMP.Model()
    map_moi_to_jump = MOI.copy_to(JuMP.backend(jump_dual_model), dual_model)
    set_optimizer(jump_dual_model, problem_iterator.optimizer)

    # Get dual variables for the parameters
    load_dual_idxs = [map_moi_to_jump[primal_dual_map.primal_parameter[l]].value for l in load_moi_idx]
    load_var_dual = JuMP.all_variables(jump_dual_model)[load_dual_idxs]

    # Add constraints to the dual associated with the parameters
    problem_iterator.set_iterator!(jump_dual_model, parameters, idx)

    # Get the objective function
    obj = objective_function(jump_dual_model)
    dual_sense = JuMP.objective_sense(jump_dual_model)

    # Inforce primal constraints
    problem_iterator.primal_builder!(jump_dual_model, load_var_dual)

    # Re-set objective function in case primal_builder! overwrote it
    @objective(jump_dual_model, dual_sense, obj)

    # Solve the dual
    JuMP.optimize!(jump_dual_model)
    optimal_loads = value.(load_var_dual)
    optimal_dual_cost = JuMP.objective_value(jump_dual_model)

    # Create final primal model and solve
    model = JuMP.Model(problem_iterator.optimizer)
    problem_iterator.primal_builder!(jump_dual_model, optimal_loads)
    JuMP.optimize!(model)

    # Check if method was effective
    optimal_final_cost = JuMP.objective_value(model)
    termination_status = JuMP.termination_status(model)
    solution_primal_status = JuMP.primal_status(model)
    solution_dual_status = JuMP.dual_status(model)
    termination_status == MOI.INFEASIBLE && @error("Optimal solution not found")
    solution_primal_status != MOI.FEASIBLE_POINT && @error("Primal solution not found")
    solution_dual_status != MOI.FEASIBLE_POINT && @error("Dual solution not found")
    
    if !isapprox(optimal_final_cost, optimal_dual_cost; rtol=1e-4)
        rtol = abs(optimal_final_cost - optimal_dual_cost) / optimal_final_cost * 100
        @warn "Final cost is not equal to dual cost by $(rtol) %" optimal_final_cost optimal_dual_cost
    end

    if recorder.filterfn(model)
        record(recorder, problem_iterator.ids[idx])
        return 1
    end
    return 0
end