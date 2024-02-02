"""

    inconvexhull(training_set::Matrix{Float64}, test_set::Matrix{Float64})

Check if new points are inside the convex hull of the given points. Solves a linear programming problem to check if the points are inside the convex hull.
"""
function inconvexhull(training_set::Matrix{Float64}, test_set::Matrix{Float64}, solver)
    # Get the number of points and dimensions
    n, d = size(training_set)
    m, d = size(test_set)
    
    # Create the model
    model = JuMP.Model(solver)
    
    # Create the variables
    @variable(model, lambda[1:n, 1:m] >= 0)
    @constraint(model, convex_combination[i=1:m], sum(lambda[j, i] for j in 1:n) == 1)
    
    # slack variables
    @variable(model, slack[1:m] >= 0)

    # Create the constraints
    @constraint(model, in_convex_hull[i=1:m, k=1:d], sum(lambda[j, i] * training_set[j, k] for j in 1:n) == test_set[i, k] + slack[i])

    # Create the objective
    @objective(model, Min, sum(slack[i] for i in 1:m))
    
    # solve the model
    optimize!(model)

    # return if the points are inside the convex hull
    return isapprox.(value.(slack), 0)
end