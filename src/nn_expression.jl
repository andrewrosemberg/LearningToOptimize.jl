using JuMP

import NNlib: relu

big_M = 1e10

function NNlib.relu(ex::AffExpr)
    model = owner_model(ex)
    aux = @variable(model, binary = true)
    relu_out = @variable(model, lower_bound = 0.0)
    @constraint(model, relu_out >= ex)
    @constraint(model, relu_out <= ex + big_M * (1 - aux))
    @constraint(model, relu_out <= big_M * aux)
    return relu_out
end

# function build_test_nlp_model()
#     model = Model()

#     @variable(model, x[i = 1:2]);

#     @variable(model, y[i = 1:2] >= 0.0);

#     ex1 = sin(x[1])
#     ex2 = cos(x[2])

#     cons = @NLconstraint(model, ex1 == ex2)

#     @objective(model, Min, sum(x) + sum(y))

#     return model, cons
# end

# function constraints_nlp_evaluator(model, x)
#     d = NLPEvaluator(model)
#     MOI.initialize(d, [:ExprGraph])

#     f = zeros(length(model.nlp_model.constraints))

#     MOI.eval_constraint(d, f, x)

#     return f
# end

# model, cons = build_test_nlp_model()

# constraints_nlp_evaluator(model, [1.0])

# ################

# flux_model = Chain(Dense(3, 3, relu), Dense(3, 1))

# ex = flux_model(x)[1]

# cons = @constraint(model, ex == 1)



# optimize!(model)

# println(value.(x))

# value(cons)

