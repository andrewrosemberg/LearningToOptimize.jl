module L2O

using Arrow
using CSV
using Dualization
using JuMP
using UUIDs
import ParametricOptInterface as POI
import JuMP.MOI as MOI
import Base: string
using Statistics

using Nonconvex
using Zygote

using MLJFlux
using Flux
using Flux: @functor
using Random
import MLJFlux.train!
using Optimisers

export ArrowFile,
    CSVFile,
    ProblemIterator,
    Recorder,
    save,
    solve_batch,
    WorstCaseProblemIterator,
    set_primal_variable!,
    set_dual_variable!,
    set_model!,
    FullyConnected,
    FullyConnectedBuilder,
    make_convex!,
    make_convex,
    ConvexRule,
    relative_rmse,
    relative_mae,
    inconvexhull

include("datasetgen.jl")
include("csvrecorder.jl")
include("arrowrecorder.jl")
include("worst_case.jl")
include("worst_case_iter.jl")
include("FullyConnected.jl")
include("nn_expression.jl")
include("metrics.jl")
include("inconvexhull.jl")

end
