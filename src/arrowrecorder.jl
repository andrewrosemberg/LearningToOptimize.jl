abstract type ArrowFile <: FileType end

Base.string(::Type{ArrowFile}) = "arrow"

"""
    record(recorder::Recorder{ArrowFile}, id::UUID)

Record optimization problem solution to an Arrow file.
"""
function record(recorder::Recorder{ArrowFile}, id::UUID; input=false)
    _filename = input ? filename_input(recorder) : filename(recorder)

    _filename = _filename * "_$(string(id))." * string(ArrowFile)

    model = if length(recorder.primal_variables) > 0
        owner_model(recorder.primal_variables[1])
    elseif length(recorder.dual_variables) > 0
        owner_model(recorder.dual_variables[1])
    else
        @error("Recorder has no variables")
    end

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
        df=merge(df, (;objective=[JuMP.objective_value(model)]))
    end
    
    return Arrow.write(
        _filename,
        df,
    )
end

function save(table::NamedTuple, filename::String, ::Type{ArrowFile})
    filename = filename * "." * string(ArrowFile)
    return Arrow.write(filename, table)
end
