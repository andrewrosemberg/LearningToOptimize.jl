using JuMP

# # Create a JuMP model object
# m = Model()
# M = 1000
# # Create binary variables
# @variable(m, x[1:2], Bin)
# # Create linear variable
# @variable(m, 0.0 <= y[1:2] <= 10.0)
# # Create linear constraint
# @constraint(m, 5 * y[1] + 3 * y[2] <= 1)
# # Create binary constraint
# @constraint(m, x[1] + x[2] <= 1)
# # Create big-M constraint
# @constraint(m, y[1] <= M * x[1])
# @constraint(m, y[2] <= M * x[2])
# # Create objective function
# @objective(m, Max, 0.8 * x[1] + 0.5 * x[2] + y[1] + y[2])


"""
    all_binary_variables(m::Model)

Return a list of all binary variables in the model `m`.
"""
function all_binary_variables(m::Model)
    return all_variables(m)[is_binary.(all_variables(m))]
end

variables(exp::VariableRef) = [exp]
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

m2, var_mapping, cons_mapping = copy_binary_model(m)
delete_binary_terms!(m)
