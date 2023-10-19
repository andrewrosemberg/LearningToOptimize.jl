mutable struct FullyConnected
    layers::PairwiseFusion
    pass_through::Vector{<:Dense}
end

@functor FullyConnected

function (model::FullyConnected)(x)
    pass_through_inputs = [layer(x) for layer in model.pass_through]
    return model.layers((x, pass_through_inputs...))[end]
end

function Base.show(io::IO, model::FullyConnected)
    println(io, "FullyConnected(")
    println(io, "     Layers: ", model.layers, ",")
    println(io,  "     Pass-Through: ", model.pass_through)
    println(io, ")")
end

"""
    FullyConnected(input_size::Int, hidden_sizes::Vector{Int}, output_size::Int)::FullyConnected

Create a fully connected neural network with `input_size` inputs, `hidden_sizes` hidden layers and `output_size` outputs.
Adds a pass through layer for each layer.
"""
function FullyConnected(input_size::Int, hidden_sizes::Vector{Int}, output_size::Int; init=Flux.glorot_uniform(Random.GLOBAL_RNG))::FullyConnected
    # Create layers
    layers = []

    # Create the pass through layers
    pass_through = [Dense(input_size, 1, init=init) for _ in 2:length(hidden_sizes)+1]

    # Create the first layer connected to the input size
    push!(layers, Dense(input_size, hidden_sizes[1], relu, init=init))

    # Create connections between hidden layers
    for i in 2:length(hidden_sizes)
        # Create a new layer
        push!(layers, Dense(hidden_sizes[i - 1] + 1, hidden_sizes[i], relu, init=init))
    end

    # Create the output layer connected to the last hidden layer
    push!(layers, Dense(hidden_sizes[end] + 1, output_size, init=init))
    
    return FullyConnected(PairwiseFusion(vcat, layers...), pass_through)
end

mutable struct FullyConnectedBuilder <: MLJFlux.Builder
	hidden_sizes::Vector{Int}
end

function MLJFlux.build(builder::FullyConnectedBuilder, rng, n_in, n_out)
	init = Flux.glorot_uniform(rng)
	return Chain(FullyConnected(n_in, builder.hidden_sizes, n_out; init=init))
end

# mutable struct ConvexRegressor <: MLJFlux.MLJFluxDeterministic
#     model::MLJFlux.Regressor
# end

# @forward((ConvexRegressor, :model), MLJFlux.Regressor)

# Define a container to hold any optimiser specific parameters (if any):
struct ConvexRule <: Flux.Optimise.AbstractOptimiser
    rule::Flux.Optimise.AbstractOptimiser
end

"""
    function make_convex!(chain::PairwiseFusion; tol = 1e-6)

Make a `PairwiseFusion` model convex by making sure all dense layers (a part from the first) have positive weights.
This procedure only makes sense for single output fully connected models.
"""
function make_convex!(chain::PairwiseFusion; tol = 1e-6)
    for layer in chain.layers[2:end]
        layer.weight .= max.(layer.weight, tol)
    end
end

"""
    function make_convex!(model::FullyConnected; tol = 1e-6)

Make a `FullyConnected` model convex by making sure all dense layers (a part from the first) have positive weights.
This procedure only makes sense for single output fully connected models.
"""
function make_convex!(model::FullyConnected; tol = 1e-6)
    make_convex!(model.layers; tol = tol)
end

function make_convex!(model::Chain; tol = 1e-6)
    for i in 1:length(model.layers)
        make_convex!(model.layers[i]; tol = tol)
    end
end

function MLJFlux.train!(model::MLJFlux.MLJFluxDeterministic, penalty, chain, optimiser::ConvexRule, X, y)
    loss = model.loss
    n_batches = length(y)
    training_loss = zero(Float32)
    for i in 1:n_batches
        parameters = Flux.params(chain)
        gs = Flux.gradient(parameters) do
            yhat = chain(X[i])
            batch_loss = loss(yhat, y[i]) + penalty(parameters) / n_batches
            training_loss += batch_loss
            return batch_loss
        end
        Flux.update!(optimiser.rule, parameters, gs)
        make_convex!(chain)
    end
    return training_loss / n_batches
end
