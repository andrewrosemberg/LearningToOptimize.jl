using TestEnv
TestEnv.activate()

using PowerModels
using PGLib
using MLJFlux
using Gurobi
using Flux
using MLJ
using L2O
using JuMP
using CUDA
using Logging

include("examples/powermodels/powermodels.jl")

matpower_case_name = "pglib_opf_case5_pjm"

network_data = make_basic_network(pglib(matpower_case_name))

branch = network_data["branch"]["1"]
f_bus_index = branch["f_bus"]
f_bus = network_data["bus"]["$f_bus_index"]
t_bus_index = branch["t_bus"]
t_bus = network_data["bus"]["$t_bus_index"]

f_owms = function_ohms_yt_from(branch)

num_samples = 1000

vm_fr, vm_to = rand(f_bus["vmin"]:0.0001:f_bus["vmax"]), rand(t_bus["vmin"]:0.0001:t_bus["vmax"])
va_fr, va_to = rand(branch["angmin"]:0.0001:branch["angmax"], num_samples), rand(branch["angmin"]:0.0001:branch["angmax"], num_samples)
a_diff = va_fr - va_to

# using Plots
f_owms_val = f_owms.(vm_fr, vm_to, va_fr, va_to)
# plt = scatter(a_diff, [i[1] for i in f_owms_val], label="p_fr", xlabel="θ_fr - θ_to", ylabel="flow", legend=:outertopright);
# scatter!(plt, a_diff, [i[2] for i in f_owms_val], label="q_fr");
optimiser=Flux.Optimise.Adam()
# Define Model
# model = MultitargetNeuralNetworkRegressor(;
#     builder=FullyConnectedBuilder([32, 64]),
#     rng=123,
#     epochs=10,
#     optimiser=optimiser,
#     acceleration=CUDALibs(),
# )

# Define the machine
_vm_fr, _vm_to = rand(f_bus["vmin"]:0.0001:f_bus["vmax"], num_samples), rand(t_bus["vmin"]:0.0001:t_bus["vmax"], num_samples)
_va_fr, _va_to = rand(branch["angmin"]:0.0001:branch["angmax"], num_samples), rand(branch["angmin"]:0.0001:branch["angmax"], num_samples)
X = [_vm_fr _vm_to _va_fr _va_to]
y = [i[1] for i in f_owms_val][:,:]

# mach = machine(model, X, y)
# fit!(mach; verbosity=2)

# Make predictions
# predictions = predict(mach, [fill(vm_fr, num_samples) fill(vm_to, num_samples) va_fr va_to])

# scatter!(plt, a_diff, predictions[:,1], label="p_fr_pred");
# scatter!(plt, a_diff, predictions[:,2], label="q_fr_pred")

loss = Flux.mse
model = FullyConnected(4, [3], 1)
opt_state = Flux.setup(optimiser, model)
best_model = model
best_loss = 1000000
for ep in 1:10
    epochloss = L2O.train!(model, loss, opt_state, X', y')
    if ep % 100 == 0
        @info("Epoch $ep, loss = $epochloss")
        if epochloss < best_loss
            best_loss = epochloss
            best_model = deepcopy(model)
        else
            model = deepcopy(best_model)
        end
    end
end


function function_ohms_yt_from(::Dict)
    return (vm_fr, vm_to, va_fr, va_to) -> mach.fitresult[1]([vm_fr, vm_to, va_fr, va_to])
end

function function_ohms_yt_to(branch::Dict)
    return (vm_fr, vm_to, va_fr, va_to) -> mach.fitresult[1]([vm_fr, vm_to, va_fr, va_to])
end

pm = instantiate_model(
    network_data,
    ACPPowerModel,
    PowerModels.build_opf;
    setting=Dict("output" => Dict("branch_flows" => true, "duals" => true)),
)

# solve
result = optimize_model!(pm, optimizer=Gurobi.Optimizer)