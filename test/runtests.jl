using L2O
using Arrow
using Test
using DelimitedFiles
using JuMP, HiGHS
import ParametricOptInterface as POI
using Flux

const test_dir = dirname(@__FILE__)
const examples_dir = joinpath(test_dir, "..", "examples")

include(joinpath(test_dir, "datasetgen.jl"))

include(joinpath(examples_dir, "powermodels", "pglib_datagen.jl"))

include(joinpath(examples_dir, "powermodels", "flux_forecaster.jl"))

@testset "L2O.jl" begin
    mktempdir() do path
        testdataset_gen(path)
        file_in, file_out = test_pglib_datasetgen(path, "pglib_opf_case5_pjm.m", 20)
        test_flux_forecaster(file_in, file_out)
    end
end
