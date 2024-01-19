###################################################
############## Unit Commitment Model ##############
###################################################

##############
# Load Functions
##############

include("examples/unitcommitment/bnb_dataset.jl")

include("src/cutting_planes.jl")

data_dir = joinpath(pwd(), "examples/unitcommitment", "data") # joinpath(dirname(@__FILE__), "data")

##############
# Load Instance
##############
case_name = "case300"
date = "2017-01-01"
save_file = case_name * "_" * replace(date, "-" => "_") * "_h" * string(instance.time)

##############
# Fit DNN approximator
##############
using MLJFlux
using CUDA
using Flux
using MLJ

# Read input and output data
input_data = CSV.read(joinpath(data_dir, input_file), DataFrame)
output_data = CSV.read(joinpath(data_dir, output_file), DataFrame)

# Separate input and output variables
output_variables = output_data[!, :objective]
input_features = innerjoin(input_data, output_data[!, [:id]], on = :id)[!, Symbol.(bin_vars_names)] # just use success solves

# optimiser=Flux.Optimise.Adam()
optimiser=ConvexRule(
    Flux.Optimise.Adam(0.01, (0.9, 0.999), 1.0e-8, IdDict{Any,Any}())
)
nn = MultitargetNeuralNetworkRegressor(;
    builder=FullyConnectedBuilder([1024, 1024, 300, 64, 32]), # [1024, 300, 64, 32]
    rng=123,
    epochs=100,
    optimiser=optimiser,
    acceleration=CUDALibs(),
    batch_size=24,
)

X = vcat(X, Matrix(input_features))
y = vcat(y, output_variables)

# Constrols

clear() = begin
    global losses = []
    global training_losses = []
    global epochs = []
    return nothing
end

function update_loss(loss)
    @info "Loss: $loss"
    push!(losses, loss)
end
update_training_loss(report) =
    push!(training_losses,
          report.training_losses[end])
update_epochs(epoch) = push!(epochs, epoch)

controls=[Step(1),
    NumberSinceBest(6),
    InvalidValue(),
    TimeLimit(5/60),
    WithLossDo(update_loss),
    WithReportDo(update_training_loss),
    WithIterationsDo(update_epochs)
]

# WIP
function SobolevLoss_mse(x, y)
    o_term = Flux.mse(x, y[:, 1])
    d_term = Flux.mse(gradient( ( _x ) -> sum(layer( _x )), x), y[:, 2:end])
    return o_term + d_term * 0.1
end

iterated_pipe =
    IteratedModel(model=nn,
        controls=controls,
        resampling=Holdout(fraction_train=0.8),
        measure = l2,
)

# Fit model
clear()
mach = machine(iterated_pipe, X, y)
fit!(mach; verbosity=2)

# Make predictions
predictions = convert.(Float64, predict(mach, X))
error = abs.(y .- predictions) ./ y
mean(error)

# Derivatives
layer = mach.fitresult.fitresult[1]
gradient( ( _x ) -> sum(layer( _x )), X')


##############
# Solve using DNN approximator
##############

function NNlib.relu(ex::AffExpr)
    model = owner_model(ex)
    relu_out = @variable(model, lower_bound = 0.0)
    @constraint(model, relu_out >= ex)
    return relu_out
end

# add nn to model
bin_vars_upper = [inner_2_upper_map[var] for var in bin_vars]
obj = mach.fitresult.fitresult[1](bin_vars_upper)

aux = @variable(upper_model)
@constraint(upper_model, obj[1] <= aux)
@constraint(upper_model, aux >= 0.0)
@objective(upper_model, Min, aux)

set_optimizer(upper_model, upper_solver)
JuMP.optimize!(upper_model)

termination_status(upper_model)
sol_bin_vars = round.(Int, value.(bin_vars_upper))
objective_value(upper_model)

##############
# Compare
##############
# Compare using approximator
mach.fitresult.fitresult[1](sol_bin_vars .* 1.0)
abs(value(obj[1]) - true_ob_value) / true_ob_value

# Compare using foward pass
for i in 1:length(bin_vars_upper)
    fix(bin_vars[i], sol_bin_vars[i])
end

JuMP.optimize!(model)
abs(objective_value(model) - true_ob_value) / true_ob_value