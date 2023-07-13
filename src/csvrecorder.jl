abstract type CSVFile <: RecorderFile end

Base.string(::Type{CSVFile}) = "csv"

"""
    record(recorder::Recorder{CSVFile}, model::JuMP.Model, id::Int64)

Record optimization problem solution to a CSV file.
"""
function record(recorder::Recorder{CSVFile}, model::JuMP.Model, id::Int64)
    if !isfile(recorder.filename)
        open(recorder.filename, "w") do f
            write(f, "id")
            for p in recorder.primal_variables
                write(f, ",$p")
            end
            for p in recorder.dual_variables
                write(f, ",dual_$p")
            end
            write(f, "\n")
        end
    end
    open(recorder.filename, "a") do f
        write(f, "$id")
        for p in recorder.primal_variables
            val = MOI.get(model, MOI.VariablePrimal(), model[p])
            write(f, ",$val")
        end
        for p in recorder.dual_variables
            val = MOI.get(model, MOI.ConstraintDual(), model[p])
            write(f, ",$val")
        end
        write(f, "\n")
    end
end