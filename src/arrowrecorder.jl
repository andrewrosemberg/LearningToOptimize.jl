abstract type ArrowFile <: FileType end

Base.string(::Type{ArrowFile}) = "arrow"

"""
    record(recorder::Recorder{ArrowFile}, id::UUID)

Record optimization problem solution to an Arrow file.
"""
function record(recorder::Recorder{ArrowFile}, id::UUID; input=false)
    _filename = input ? filename_input(recorder) : filename(recorder)

    _filename = _filename * "_$(string(id))." * string(ArrowFile)

    model = recorder.model

    df = (;
        id=[id],
        zip(
            Symbol.(name.(recorder.primal_variables)),
            [[value.(p)] for p in recorder.primal_variables],
        )...,
        zip(
            Symbol.("dual_" .* name.(recorder.dual_variables)),
            [[dual.(p)] for p in recorder.dual_variables],
        )...,
    )
    if !input
        df = merge(
            df,
            (;
                objective=[JuMP.objective_value(model)],
                time=[JuMP.solve_time(model)],
                status=[string(JuMP.termination_status(model))],
                primal_status=[string(JuMP.primal_status(model))],
                dual_status=[string(JuMP.dual_status(model))],
            ),
        )
    end

    return Arrow.write(_filename, df)
end

function save(table::NamedTuple, filename::String, ::Type{ArrowFile})
    filename = filename * "." * string(ArrowFile)
    return Arrow.write(filename, table)
end
