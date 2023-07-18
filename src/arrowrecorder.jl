abstract type ArrowFile <: RecorderFile end

Base.string(::Type{ArrowFile}) = "arrow"

"""
    record(recorder::Recorder{ArrowFile}, id::UUID)

Record optimization problem solution to an Arrow file.
"""
function record(recorder::Recorder{ArrowFile}, id::UUID)
    return Arrow.append(
        recorder.filename,
        (;
            id=[id],
            zip(
                Symbol.(name.(recorder.primal_variables)),
                [[value.(p)] for p in recorder.primal_variables],
            )...,
            zip(
                Symbol.("dual_" .* name.(recorder.dual_variables)),
                [[dual.(p)] for p in recorder.dual_variables],
            )...,
        ),
    )
end

function save(table::NamedTuple, filename::String, ::Type{ArrowFile})
    return Arrow.write(filename, table)
end
