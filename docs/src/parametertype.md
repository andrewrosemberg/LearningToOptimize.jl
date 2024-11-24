
### **Parameter Types for Optimization Problems**

#### **Introduction**
When working with optimization problems, `LearningToOptimize.jl` supports multiple parameterization strategies. These strategies influence the behavior and performance of the solver and allow flexibility in retrieving information such as dual variables.

---

#### **Comparison**

| **Parameter Type**      | **Supported Problems** | **Dual Support** | **Performance** | **Notes**                        |
|--------------------------|------------------------|------------------|-----------------|----------------------------------|
| `JuMPParameterType`      | All                   | Yes              | Moderate        | General-purpose; slower.        |
| `JuMPNLPParameterType`   | NLP Only              | No               | Fast            | Optimized for NLP problems.      |
| `POIParameterType` (Default) | Linear, Conic         | Yes              | Fast            | Requires POI-wrapped solvers.    |



---

#### **Supported Parameter Types**

1. **`POIParameterType` (Default)**:
   - **Description**:
     - Extends MOI.Parameters for linear and conic problems.
     - Compatible with solvers wrapped using `ParametricOptInterface`.
     - Supports fetching duals w.r.t. parameters.
   - **Limitations**:
     - Not compatible with nonlinear solvers.
   - **Usage Example**:
     Default behavior when using `ProblemIterator` without specifying `param_type`.

2. **`JuMPParameterType`**:
   - **Description**:
     - Adds a variable as a parameter with an additional constraint during `solve_batch`.
     - Slower compared to other types but supports fetching duals w.r.t. the parameter.
     - Compatible with all problem types.
   - **Usage Example**:
     ```julia
     using JuMP, LearningToOptimize

     model = JuMP.Model(HiGHS.Optimizer)
     @variable(model, x)
     p = @variable(model, _p)
     @constraint(model, cons, x + _p >= 3)
     @objective(model, Min, 2x)

     num_p = 10
     problem_iterator = ProblemIterator(
         Dict(p => collect(1.0:num_p));
         param_type = LearningToOptimize.JuMPParameterType
     )

     recorder = Recorder{ArrowFile}("output.arrow"; primal_variables = [x], dual_variables = [cons])
     solve_batch(problem_iterator, recorder)
     ```
   - **Advantages**:
     - Works with all solvers and problem types.
     - Duals w.r.t. parameters are available.

3. **`JuMPNLPParameterType`**:
   - **Description**:
     - Utilizes MOI’s internal parameter structure.
     - Optimized for speed but limited to nonlinear programming (NLP) problems.
     - Does not support fetching duals w.r.t. parameters.
   - **Usage Example**:
     ```julia
     using JuMP, LearningToOptimize

     model = JuMP.Model(Ipopt.Optimizer)
     @variable(model, x)
     p = @variable(model, _p in MOI.Parameter(1.0))
     @constraint(model, cons, x + _p >= 3)
     @objective(model, Min, 2x)

     num_p = 10
     problem_iterator = ProblemIterator(
         Dict(p => collect(1.0:num_p));
         param_type = LearningToOptimize.JuMPNLPParameterType
     )

     recorder = Recorder{ArrowFile}("output.arrow"; primal_variables = [x], dual_variables = [cons])
     solve_batch(problem_iterator, recorder)
     ```
   - **Advantages**:
     - Fast and efficient for NLP problems.
     - No external wrappers required.
