module L2O

using Arrow
using JuMP
import ParametricOptInterface as POI
import Base: string

export ArrowFile, CSVFile, ProblemIterator, Recorder, solve_batch


include("datasetgen.jl")
include("csvrecorder.jl")
include("arrowrecorder.jl")

end
