module L2O

using Arrow
using CSV
using Dualization
using JuMP
using UUIDs
import ParametricOptInterface as POI
import JuMP.MOI as MOI
import Base: string

using Nonconvex
using Zygote

export ArrowFile, CSVFile, ProblemIterator, Recorder, save, solve_batch, 
    WorstCaseProblemIterator, set_primal_variable!, set_dual_variable!

include("datasetgen.jl")
include("csvrecorder.jl")
include("arrowrecorder.jl")
include("worst_case.jl")
include("worst_case_iter.jl")

end
