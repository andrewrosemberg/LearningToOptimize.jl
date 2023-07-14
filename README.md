# L2O.jl
Learning to optimize (L2O) package that provides basic functionalities to help fit proxy models for optimization.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://andrewrosemberg.github.io/L2O.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://andrewrosemberg.github.io/L2O.jl/dev/)
[![Build Status](https://github.com/andrewrosemberg/L2O.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/andrewrosemberg/L2O.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/andrewrosemberg/L2O.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/andrewrosemberg/L2O.jl)

## Generate Dataset
This package provides a basic way of generating a dataset of the solutions of an optimization problem by varying the values of the parameters in the problem and recording it.

The user needs to first define a problem iterator:

```julia
# The problem to iterate over
model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
@variable(model, x)
p = @variable(model, _p in POI.Parameter(1.0)) # The parameter (defined using POI)
@constraint(model, cons, x + _p >= 3)
@objective(model, Min, 2x)

# The ids
problem_ids = collect(1:10)

# The parameter values
parameter_values = Dict(p => collect(1.0:10.0))

# The iterator
problem_iterator = ProblemIterator(problem_ids, parameter_values)
```

Then chose a type of recorder and what values to record:

```julia
# CSV recorder to save the optimal primal and dual decision values
recorder = Recorder{CSVFile}("test.csv", primal_variables=[:x], dual_variables=[:cons])

# Finally solve all problems described by the iterator
solve_batch(model, problem_iterator, recorder)
```

Which creates the following CSV:

| id |   x  | dual_cons |
|----|------|-----------|
|  1 |  2.0 |       2.0 |
|  2 |  1.0 |       2.0 |
|  3 | -0.0 |       2.0 |
|  4 | -1.0 |       2.0 |
|  5 | -2.0 |       2.0 |
|  6 | -3.0 |       2.0 |
|  7 | -4.0 |       2.0 |
|  8 | -5.0 |       2.0 |
|  9 | -6.0 |       2.0 |
| 10 | -7.0 |       2.0 |

Similarly, there is also the option to save the database in arrow files:

```julia
recorder = Recorder{ArrowFile}("test.arrow", primal_variables=[:x], dual_variables=[:cons])
```
