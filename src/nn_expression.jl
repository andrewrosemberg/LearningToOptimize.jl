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


