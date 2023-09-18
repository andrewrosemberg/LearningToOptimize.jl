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

const test_dir = dirname(@__FILE__)
const examples_dir = joinpath(test_dir, "..", "examples")

include(joinpath(test_dir, "datasetgen.jl"))

include(joinpath(test_dir, "worst_case.jl"))

include(joinpath(examples_dir, "powermodels", "pglib_datagen.jl"))

include(joinpath(examples_dir, "flux", "test_flux_forecaster.jl"))

@testset "L2O.jl" begin
    mktempdir() do path
        test_problem_iterator(path)
        test_worst_case_problem_iterator(path)
        file_in, file_out = test_pglib_datasetgen(path, "pglib_opf_case5_pjm", 20)
        file_in, file_out = test_generate_worst_case_dataset(path, "pglib_opf_case5_pjm", 20)
        test_flux_forecaster(file_in, file_out)
    end
end
