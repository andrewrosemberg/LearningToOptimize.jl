using JuMP
import ParametricOptInterface as POI
using HiGHS
using Flux

"""
    simulate_multistage(
        subproblems::Vector{JuMP.Model},
        state_params_in::Vector{Vector{VariableRef}},
        state_params_out::Vector{Vector{VariableRef}},
        initial_state::Vector{Float64},
        decision_rules::Vector{Function},
        Uncertainties::Vector{Dict{VariableRef, Float64}},
    )

Simulate a multistage optimization problem.

# Arguments
- `subproblems::Vector{JuMP.Model}`: Vector of subproblems
- `state_params_in::Vector{Vector{VariableRef}}`: Vector of state parameters for each stage
- `state_params_out::Vector{Vector{VariableRef}}`: Vector of state parameters for each stage
- `initial_state::Vector{Float64}`: Initial state
- `decision_rules::Vector{Function}`: Vector of decision rules for each stage
- `Uncertainties::Vector{Dict{VariableRef, Float64}}`: Vector of uncertainties for each stage
"""
function simulate_multistage(
    subproblems::Vector{JuMP.Model},
    state_params_in::Vector{Vector{VariableRef}},
    state_params_out::Vector{Vector{VariableRef}},
    initial_state::Vector{Float64},
    decision_rules::Vector{F},
    uncertainties::Vector{Dict{VariableRef, Float64}},
    ) where {F}
    
    # Initialize state
    state = initial_state
    
    # Loop over stages
    objective_value = 0.0
    for stage in 1:length(subproblems)
        
        # Get subproblem
        subproblem = subproblems[stage]
        
        # Update state parameters
        state_param_in = state_params_in[stage]
        for (i, state_var) in enumerate(state_param_in)
            MOI.set(subproblem, POI.ParameterValue(), state_var, state[i])
        end
        
        # Update uncertainty
        uncertainties_stage = uncertainties[stage]
        for (uncertainty_param, uncertainty_value) in uncertainties_stage
            MOI.set(subproblem, POI.ParameterValue(), uncertainty_param, uncertainty_value)
        end
        
        # Get decisions and transition state
        decision_rule = decision_rules[stage]
        state = decision_rule([state; collect(values(uncertainties_stage))])
        
        # Update state parameters out
        state_param_out = state_params_out[stage]
        for (i, state_var) in enumerate(state_param_out)
            MOI.set(subproblem, POI.ParameterValue(), state_var, state[i])
        end
        
        # Solve subproblem
        optimize!(subproblem)

        # Update objective value
        objective_value += JuMP.objective_value(subproblem) 
    end
    
    # Return final objective value
    return objective_value
end
