##################################################
######### Unit Commitment Proxy Training #########
##################################################

##############
# Load Functions
##############

import Pkg; Pkg.activate(dirname(dirname(@__DIR__))); Pkg.instantiate()

using L2O
using MLJFlux
using CUDA # if error run CUDA.set_runtime_version!(v"12.1.0")
using Flux
using MLJ
using DataFrames

include(joinpath(dirname(@__FILE__), "bnb_dataset.jl"))

include(joinpath(dirname(dirname(@__DIR__)), "src/cutting_planes.jl"))

data_dir = joinpath(dirname(@__FILE__), "data")

##############
# Parameters
##############
filetype=ArrowFile
case_name = ARGS[3] # case_name = "case300"
date = ARGS[4] # date="2017-01-01"
horizon = parse(Int, ARGS[5]) # horizon=2
save_file = case_name * "_" * replace(date, "-" => "_") * "_h" * string(horizon)
data_dir = joinpath(data_dir, case_name, date, "h" * string(horizon))

##############
# Fit DNN approximator
##############

# Read input and output data
iter_input = readdir(joinpath(data_dir, "input"), join=true)
iter_output = readdir(joinpath(data_dir, "output"), join=true)
# filter for only arrow files of this case
iter_input = [file for file in iter_input if occursin(case_name, file) && occursin("arrow", file)]
iter_output = [file for file in iter_output if occursin(case_name, file) && occursin("arrow", file)]

# Load input and output data tables
input_tables = Array{DataFrame}(undef, length(iter_input))
output_tables = Array{DataFrame}(undef, length(iter_output))
for (i, file) in enumerate(iter_input)
    input_tables[i] = Arrow.Table(file) |> DataFrame
end
for (i, file) in enumerate(iter_output)
    output_tables[i] = Arrow.Table(file) |> DataFrame
end

# concatenate all the input and output tables
input_table = vcat(input_tables...)
output_table = vcat(output_tables...)

# Separate input and output variables & ignore id time status primal_status dual_status
y = output_table[!, :objective]
input_features = innerjoin(input_table, output_table[!, [:id]], on = :id)[!, Not(:id)] # just use success solves
X = Matrix(input_features)

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
# function SobolevLoss_mse(x, y)
#     o_term = Flux.mse(x, y[:, 1])
#     d_term = Flux.mse(gradient( ( _x ) -> sum(layer( _x )), x), y[:, 2:end])
#     return o_term + d_term * 0.1
# end

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