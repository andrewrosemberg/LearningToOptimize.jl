"""

    inconvexhull(training_set::Matrix{Float64}, test_set::Matrix{Float64})

Check if new points are inside the convex hull of the given points. Solves a linear programming problem to check if the points are inside the convex hull.
"""
function inconvexhull(training_set::Matrix{Float64}, test_set::Matrix{Float64}, solver)
    # Get the number of points and dimensions
    n, d = size(training_set)
    m, d_ = size(test_set)
    @assert d == d_ "The dimensions of the training and test sets must be the same"
    
    # Create the POI model
    model = JuMP.Model(() -> POI.Optimizer(solver()))
    
    # Create the variables
    @variable(model, lambda[1:n] >= 0)
    @constraint(model, convex_combination, sum(lambda) == 1)

    # Create POI parameters
    @variable(model, test_set_params[1:d] in MOI.Parameter.(0.0))
    
    # slack variables
    @variable(model, slack[1:d])
    @variable(model, abs_slack[1:d] >= 0)
    @constraint(model, abs_slack .>= slack)
    @constraint(model, abs_slack .>= -slack)

    # Create the constraints
    @constraint(model, in_convex_hull, training_set'lambda .== test_set_params .+ slack)

    # Create the objective
    @objective(model, Min, sum(abs_slack))

    in_convex_hull = Vector{Bool}(undef, m)
    for i in 1:m
        # Set the test set parameters
        MOI.set.(model, POI.ParameterValue(), test_set_params, test_set[i, :])

        # solve the model
        optimize!(model)

        # return if the points are inside the convex hull
        in_convex_hull[i] = isapprox(JuMP.objective_value(model), 0.0)
    end
    
    return in_convex_hull
end