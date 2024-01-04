using JuMP

"""
    all_binary_variables(m::Model)

Return a list of all binary variables in the model `m`.
"""
function all_binary_variables(m::Model)
    return all_variables(m)[is_binary.(all_variables(m))]
end

variables(exp::VariableRef) = [exp]
variables(exp::Vector{VariableRef}) = exp
variables(exp::AffExpr) = [var for var in keys(exp.terms)]

function variables(con::ConstraintRef)
    con_exp = constraint_object(con).func
    return variables(con_exp)
end

"""
    all_binary_constraints(m::Model)

Return a list of all constraints in the model `m` 
containing only binary variables.
"""
function all_binary_constraints(m::Model)
    all_cons_types = list_of_constraint_types(m)
    consrefs = []
    for con_type in all_cons_types
        cons = all_constraints(m, con_type...)
        for con in cons
            if all([is_binary(var) for var in variables(con)])
                push!(consrefs, con)
            end
        end
    end
    return consrefs
end

# Copy variables
function copy_binary_variables(m_to::Model, m_from::Model)
    mapping = Dict()
    for var in all_binary_variables(m_from)
        ref_to = @variable(m_to, base_name = name(var), binary = true)
        mapping[var] = ref_to
    end
    return mapping
end

# Copy constraints
function copy_binary_constraint(::Model, ::VariableRef, set, var_mapping)
    return nothing
end
function copy_binary_constraint(m_to::Model, func::AffExpr, set, var_mapping)
    terms_dict = JuMP.OrderedCollections.OrderedDict{VariableRef, Float64}()
    for var in keys(func.terms)
        terms_dict[var_mapping[var]] = func.terms[var]
    end
    exp = GenericAffExpr(func.constant, terms_dict)
    return @constraint(m_to, exp in set)
end

function copy_binary_constraints(m_to::Model, m_from::Model, var_mapping::Dict)
    cons_to = []
    for con in all_binary_constraints(m_from)
        con_exp = constraint_object(con)
        ref_to = copy_binary_constraint(m_to, con_exp.func, con_exp.set, var_mapping)
        push!(cons_to, ref_to)
    end
    return cons_to
end

# Copy objective function
function copy_binary_objective(m_to::Model, m_from::Model, var_mapping::Dict)
    obj = objective_function(m_from)
    obj2_dict = JuMP.OrderedCollections.OrderedDict{VariableRef, Float64}()
    for var in keys(obj.terms)
        if is_binary(var)
            obj2_dict[var_mapping[var]] = obj.terms[var]
        end
    end
    obj2 = GenericAffExpr(obj.constant, obj2_dict)
    @objective(m_to, objective_sense(m_from), obj2)
end

# Copy binary model
function copy_binary_model(m_from::Model)
    m_to = Model()
    var_mapping = copy_binary_variables(m_to, m_from)
    cons_mapping = copy_binary_constraints(m_to, m_from, var_mapping)
    copy_binary_objective(m_to, m_from, var_mapping)
    return m_to, var_mapping, cons_mapping
end

# remove binary terms
function delete_binary_terms!(m::Model)
    obj = objective_function(m)
    for var in keys(obj.terms)
        if is_binary(var)
            println("deleting $var")
            delete!(obj.terms, var)
        end
    end
    obj.constant = 0.0
    set_objective_function(m, obj)

    # remove binary constraints from the original model
    for con in all_binary_constraints(m)
        delete(m, con)
    end
end

function add_deficit_constraints!(model::Model; penalty=1e7)
    consrefs = [con for con in all_constraints(model, include_variable_in_set_constraints=false)]
    @variable(model, deficit[1:length(consrefs)])
    @variable(model, norm_deficit)
    for (i, eq) in enumerate(consrefs)
        set_normalized_coefficient(eq, deficit[i], 1)
    end
    @constraint(model, [norm_deficit; deficit] in MOI.NormOneCone(1 + length(deficit)))
    set_objective_coefficient(model, norm_deficit, penalty)
    return norm_deficit
end

# fix binary variables to POI parameters
function fix_binary_variables!(inner_model::Model, inner_2_upper_map::Dict)
    var_mapping = Dict()
    for (to_var, from_var) in inner_2_upper_map
        param = @variable(inner_model, set = MOI.Parameter(0.0))
        @constraint(inner_model, to_var == param)
        set_upper_bound(to_var, 1.1)
        set_lower_bound(to_var, -0.1)
        var_mapping[from_var] = param
    end
    # QUESTION: DOES values() and keys() return the same order?
    return var_mapping
end

# set binary variables
function set_binary_variables!(inner_model::Model, var_mapping::Dict, vals)
    for (i, to_var) in enumerate(values(var_mapping))
        MOI.set(inner_model, POI.ParameterValue(), to_var, vals[i])
    end
end

# add cut
function add_cut!(upper_model::Model, cut_intercept, cut_slope, cut_point, u)
    @constraint(upper_model, 
        upper_model[:θ] >= cut_intercept + dot(cut_slope, u .- cut_point)
    )
end

function cutting_planes!(inner_model::Model; upper_solver, inner_solver, max_iter::Int=1000)
    upper_model, inner_2_upper_map, cons_mapping = copy_binary_model(inner_model)
    delete_binary_terms!(inner_model)
    add_deficit_constraints!(inner_model)
    upper_2_inner = fix_binary_variables!(inner_model, inner_2_upper_map)
    u = keys(upper_2_inner)
    u_inner = values(upper_2_inner)
    set_optimizer(inner_model, inner_solver)
    set_optimizer(upper_model, upper_solver)

    # cut list
    JuMP.optimize!(inner_model)
    cuts_intercept = [objective_value(inner_model)]
    cuts_slope = [[MOI.get(inner_model, POI.ParameterDual(), u_i) for u_i in u_inner]]
    cuts_point = [zeros(length(upper_2_inner))]

    # cutting planes epigraph variable
    bound = -1e7
    @variable(upper_model, θ >= bound)
    obj_upper = objective_function(upper_model)
    @objective(upper_model, Min, obj_upper + θ)

    # loop
    i = 1
    gap = Array{Float64}(undef, max_iter)
    upper_bound = Array{Float64}(undef, max_iter)
    lower_bound = Array{Float64}(undef, max_iter)
    while i <= max_iter
        # Add cuts
        add_cut!(upper_model, cuts_intercept[i], cuts_slope[i], cuts_point[i], u)
    
        JuMP.optimize!(upper_model)
    
        # Add point to the lists
        if termination_status(upper_model) == MOI.OPTIMAL
            push!(cuts_point, value.(keys(upper_2_inner)))
            bound = objective_value(upper_model)
        else
            println("Upper problem failed")
            break;
        end
    
        # run inner problem
        set_binary_variables!(inner_model, upper_2_inner, cuts_point[i+1])
        JuMP.optimize!(inner_model)
    
        # Add cut to the lists
        if termination_status(inner_model) == MOI.OPTIMAL || termination_status(inner_model) == MOI.LOCALLY_SOLVED
            push!(cuts_intercept, objective_value(inner_model))
            push!(cuts_slope, [MOI.get(inner_model, POI.ParameterDual(), u_i) for u_i in u_inner])
        else
            println("Inner problem failed")
            break;
        end
    
        # test convergence
        u_bound = minimum(cuts_intercept)
        upper_bound[i] = u_bound
        lower_bound[i] = bound
        gap[i] = abs(bound - u_bound) / u_bound
        if i > 10 && gap[i] < 0.1 && all([all(cuts_point[i] .== cuts_point[j]) for j in i-10:i-1])
            println("Converged")
            break;
        else
            @info "Iteration $i" bound cuts_intercept[i]
        end
        i += 1
    end
    i = ifelse(i >= max_iter, max_iter, i)

    gap = gap[1:i]
    upper_bound = upper_bound[1:i]
    lower_bound = lower_bound[1:i]
    return upper_model, lower_bound, upper_bound, gap
end

