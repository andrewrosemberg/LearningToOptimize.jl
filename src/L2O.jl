module L2O

using Arrow
using CSV
using DataFrames
using JuMP
using UUIDs
import ParametricOptInterface as POI
import JuMP.MOI as MOI
import Base: string
using Statistics
using Distributions

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
    load,
    solve_batch,
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
    inconvexhull,
    line_sampler,
    box_sampler,
    scaled_distribution_sampler,
    general_sampler,
    compress_batch_arrow

include("datasetgen.jl")
include("csvrecorder.jl")
include("arrowrecorder.jl")
include("FullyConnected.jl")
include("nn_expression.jl")
include("metrics.jl")
include("inconvexhull.jl")
include("samplers.jl")

end
