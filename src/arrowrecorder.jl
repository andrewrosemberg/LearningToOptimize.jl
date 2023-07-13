abstract type ArrowFile <: RecorderFile end

Base.string(::Type{ArrowFile}) = "arrow"

"""
    record(recorder::Recorder{ArrowFile}, model::JuMP.Model, id::Int64)

Record optimization problem solution to an Arrow file.
"""
function record(recorder::Recorder{ArrowFile}, model::JuMP.Model, id::Int64)
    if !isfile(recorder.filename)
        ### NOT WORKING ###
        Arrow.write(recorder.filename, (id = Int64[], recorder.primal_variables..., "dual_" .* recorder.dual_variables..., ))
    end

    Arrow.write(Arrow.Table(; id = [id], [MOI.get(model, MOI.VariablePrimal(), model[p]) for p in recorder.primal_variables]..., [MOI.get(model, MOI.ConstraintDual(), model[p]) for p in recorder.dual_variables]...) |> Arrow.Table, Arrow.append!(recorder.filename))
end