using Arrow
using DelimitedFiles
using Flux
using HiGHS
using JuMP
using L2O
import ParametricOptInterface as POI
using Test
using UUIDs
using Ipopt
using MLJFlux
using Flux
using MLJ
using CSV
using DataFrames
using Optimisers

const test_dir = dirname(@__FILE__)
const examples_dir = joinpath(test_dir, "..", "examples")

include(joinpath(test_dir, "datasetgen.jl"))

include(joinpath(examples_dir, "powermodels", "pglib_datagen.jl"))

include(joinpath(test_dir, "test_flux_forecaster.jl"))

include(joinpath(test_dir, "nn_expression.jl"))

include(joinpath(test_dir, "inconvexhull.jl"))

include(joinpath(test_dir, "samplers.jl"))

@testset "L2O.jl" begin
    test_load_parameters_model()
    test_load_parameters()
    test_line_sampler()
    test_box_sampler()
    test_general_sampler()
    test_fully_connected()
    test_flux_jump_basic()
    test_inconvexhull()

    mktempdir() do path
        test_general_sampler_file(; cache_dir=path)
        test_problem_iterator(path)
        file_in, file_out = test_pglib_datasetgen(path, "pglib_opf_case5_pjm", 20)
        test_flux_forecaster(file_in, file_out)
    end
end
