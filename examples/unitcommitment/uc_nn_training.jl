##################################################
######### Unit Commitment Proxy Training #########
##################################################

##############
# Load Functions
##############

import Pkg; Pkg.activate(dirname(dirname(@__DIR__))); Pkg.instantiate()

using L2O
using MLJFlux
using CUDA
using Flux
using MLJ

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