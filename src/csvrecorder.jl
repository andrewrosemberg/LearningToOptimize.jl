abstract type CSVFile <: FileType end

Base.string(::Type{CSVFile}) = "csv"

"""
    record(recorder::Recorder{CSVFile}, id::UUID)

Record optimization problem solution to a CSV file.
"""
function record(recorder::Recorder{CSVFile}, id::UUID; input=false)
    _filename = input ? filename_input(recorder) : filename(recorder)
    _filename = _filename * "." * string(CSVFile)

    model = recorder.model
    primal_status=JuMP.primal_status(model)
    dual_status=JuMP.dual_status(model)

    if !isfile(_filename)
        open(_filename, "w") do f
            write(f, "id")
            for p in recorder.primal_variables
                write(f, ",$(name(p))")
            end
            for p in recorder.dual_variables
                write(f, ",dual_$(name(p))")
            end
            if !input
                write(f, ",objective")
                write(f, ",time")
                write(f, ",status")
                write(f, ",primal_status")
                write(f, ",dual_status")
            end
            write(f, "\n")
        end
    end
    open(_filename, "a") do f
        write(f, "$id")
        if in(primal_status, DECISION_STATUS)
            for p in recorder.primal_variables
                val = value.(p)
                write(f, ",$val")
            end
        else
            for p in recorder.primal_variables
                write(f, ",0")
            end
        end
        if in(dual_status, DECISION_STATUS)
            for p in recorder.dual_variables
                val = dual.(p)
                write(f, ",$val")
            end
        else
            for p in recorder.dual_variables
                write(f, ",0")
            end
        end

        if !input
            # save objective value
            obj = JuMP.objective_value(model)
            write(f, ",$obj")
            # save solve time
            time = JuMP.solve_time(model)
            write(f, ",$time")
            # save status
            status = JuMP.termination_status(model)
            write(f, ",$status")
            # save primal status
            write(f, ",$primal_status")
            # save dual status
            write(f, ",$dual_status")
        end
        # end line
        write(f, "\n")
    end
end

function save(table::NamedTuple, filename::String, ::Type{CSVFile}; kwargs...)
    filename = filename * "." * string(CSVFile)
    isappend = isfile(filename)
    mode = isappend ? "append" : "write"
    @info "Saving CSV file to $filename - Mode: $mode"
    CSV.write(filename, table; append=isappend)
    return nothing
end

function load(filename::String, ::Type{CSVFile})
    filename = filename * "." * string(CSVFile)
    return CSV.read(filename, DataFrame)
end
