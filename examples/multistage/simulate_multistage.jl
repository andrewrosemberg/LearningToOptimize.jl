using JuMP
import ParametricOptInterface as POI
using HiGHS
using Flux
using Zygote
using ChainRulesCore
import ChainRulesCore.rrule

function simulate_states(
    initial_state::Vector{Float64},
    uncertainties::Vector{Dict{VariableRef, Float64}},
    decision_rules::Vector{F},
) where {F}
    num_stages = length(uncertainties)
    states = Vector{Vector{Float64}}(undef, num_stages + 1)
    states[1] = initial_state
    for stage in 1:num_stages
        uncertainties_stage = uncertainties[stage]
        decision_rule = decision_rules[stage]
        states[stage + 1] = decision_rule([states[stage]; collect(values(uncertainties_stage))])
    end
    return states
end

function simulate_multistage(
    subproblems::Vector{JuMP.Model},
    state_params_in::Vector{Vector{VariableRef}},
    state_params_out::Vector{Vector{VariableRef}},
    states::Vector{Vector{Float64}},
    uncertainties::Vector{Dict{VariableRef, Float64}},
    )
    
    # Loop over stages
    objective_value = 0.0
    for stage in 1:length(subproblems)
        state_in = states[stage]
        state_out = states[stage + 1]
        # Get subproblem
        subproblem = subproblems[stage]
        
        # Update state parameters
        state_param_in = state_params_in[stage]
        for (i, state_var) in enumerate(state_param_in)
            MOI.set(subproblem, POI.ParameterValue(), state_var, state_in[i])
        end
        
        # Update uncertainty
        uncertainties_stage = uncertainties[stage]
        for (uncertainty_param, uncertainty_value) in uncertainties_stage
            MOI.set(subproblem, POI.ParameterValue(), uncertainty_param, uncertainty_value)
        end
        
        # Update state parameters out
        state_param_out = state_params_out[stage]
        for (i, state_var) in enumerate(state_param_out)
            MOI.set(subproblem, POI.ParameterValue(), state_var, state_out[i])
        end
        
        # Solve subproblem
        optimize!(subproblem)

        # Update objective value
        objective_value += JuMP.objective_value(subproblem) 
    end
    
    # Return final objective value
    return objective_value
end

pdual(v::VariableRef) = MOI.get(JuMP.owner_model(v), POI.ParameterDual(), v)
pdual(vs::Vector{VariableRef}) = [pdual(v) for v in vs]

# Define rrule of simulate_multistage
function rrule(::typeof(simulate_multistage), subproblems, state_params_in, state_params_out, states, uncertainties)
    y = simulate_multistage(subproblems, state_params_in, state_params_out, states, uncertainties)
    function _pullback(Δy)
        pull_back_model = pdual.(state_params_in).+pdual.(state_params_out) * Δy
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), pull_back_model, NoTangent())
    end
    return y, _pullback
end

function sample(uncertainty_samples::Dict{VariableRef, Vector{Float64}})
    return Dict((k => v[rand(1:end)]) for (k, v) in uncertainty_samples)
end

sample(uncertainty_samples::Vector{Dict{VariableRef, Vector{Float64}}}) = [sample(uncertainty_samples[t]) for t in 1:length(uncertainty_samples)]

function train_multistage(model, initial_state, subproblems, state_params_in, state_params_out, uncertainty_samples; num_train_samples=100, optimizer=Flux.Adam(0.01))
    num_stages = length(subproblems)
    # Initialise the optimiser for this model:
    opt_state = Flux.setup(optimizer, model)

    for _ in 1:num_train_samples
        # Sample uncertainties
        uncertainty_sample = sample(uncertainty_samples)

        # Calculate the gradient of the objective
        # with respect to the parameters within the model:
        models = Vector{Any}(undef, num_stages)
        grads = Flux.gradient(model) do m
            for (i, subproblem) in enumerate(subproblems)
                models[i] = m
            end
            states = simulate_states(
                initial_state, uncertainty_sample, 
                models
            )
            simulate_multistage(
                subproblems, state_params_in, state_params_out, states, 
                uncertainty_sample
            )
        end

        # Update the parameters so as to reduce the objective,
        # according the chosen optimisation rule:
        Flux.update!(opt_state, model, grads[1])
    end
    
    return model
end
