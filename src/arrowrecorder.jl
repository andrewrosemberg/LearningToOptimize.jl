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

    status=JuMP.termination_status(model)
    primal_stat=JuMP.primal_status(model)
    dual_stat=JuMP.dual_status(model)

    primal_values = if in(primal_stat, DECISION_STATUS)
        [[value.(p)] for p in recorder.primal_variables]
    else
        [[zeros(length(p))] for p in recorder.primal_variables]
    end

    dual_values = if in(dual_stat, DECISION_STATUS)
        [[dual.(p)] for p in recorder.dual_variables]
    else
        [[zeros(length(p))] for p in recorder.dual_variables]
    end

    objective = if in(status, ACCEPTED_TERMINATION_STATUSES)
        JuMP.objective_value(model)
    else
        0.0
    end

    df = (;
        id=[id],
        zip(
            Symbol.(name.(recorder.primal_variables)),
            primal_values,
        )...,
        zip(
            Symbol.("dual_" .* name.(recorder.dual_variables)),
            dual_values,
        )...,
    )
    if !input
        df = merge(
            df,
            (;
                objective=[objective],
                time=[JuMP.solve_time(model)],
                status=[string(status)],
                primal_status=[string(primal_stat)],
                dual_status=[string(dual_stat)],
            ),
        )
    end

    return Arrow.write(_filename, df)
end

function save(table::NamedTuple, filename::String, ::Type{ArrowFile})
    filename = filename * "." * string(ArrowFile)
    return Arrow.write(filename, table)
end

function load(filename::String, ::Type{ArrowFile})
    return DataFrame(Arrow.Table(filename * "." * string(ArrowFile)))
end
