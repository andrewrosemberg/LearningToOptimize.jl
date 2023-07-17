abstract type ArrowFile <: RecorderFile end

Base.string(::Type{ArrowFile}) = "arrow"

"""
    record(recorder::Recorder{ArrowFile}, model::JuMP.Model, id::T)

Record optimization problem solution to an Arrow file.
"""
function record(recorder::Recorder{ArrowFile}, model::JuMP.Model, id::T) where {T<:Integer}
    return Arrow.append(
        recorder.filename,
        (;
            id=[id],
            zip(
                Symbol.(name.(recorder.primal_variables)),
                [
                    [value.(p)] for
                    p in recorder.primal_variables
                ],
            )...,
            zip(
                Symbol.("dual_" .* name.(recorder.dual_variables)),
                [
                    [dual.(p)] for
                    p in recorder.dual_variables
                ],
            )...,
        ),
    )
end
