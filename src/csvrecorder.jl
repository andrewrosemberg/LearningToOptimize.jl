abstract type CSVFile <: FileType end

Base.string(::Type{CSVFile}) = "csv"

"""
    record(recorder::Recorder{CSVFile}, id::UUID)

Record optimization problem solution to a CSV file.
"""
function record(recorder::Recorder{CSVFile}, id::UUID; input=false)
    _filename = input ? filename_input(recorder) : filename(recorder)
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
            end
            write(f, "\n")
        end
    end
    open(_filename, "a") do f
        write(f, "$id")
        for p in recorder.primal_variables
            val = value.(p)
            write(f, ",$val")
        end
        for p in recorder.dual_variables
            val = dual.(p)
            write(f, ",$val")
        end
        # save objective value
        model = if length(recorder.primal_variables) > 0
            owner_model(recorder.primal_variables[1])
        elseif length(recorder.dual_variables) > 0
            owner_model(recorder.dual_variables[1])
        else
            @error("Recorder has no variables")
        end
        if !input
            obj = JuMP.objective_value(model)
            write(f, ",$obj")
        end
        # end line
        write(f, "\n")
    end
end

function save(table::NamedTuple, filename::String, ::Type{CSVFile})
    return CSV.write(filename, table)
end
