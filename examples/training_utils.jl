

function update_loss(loss)
    Wandb.log(lg, Dict("metrics/loss" => loss))
    return nothing
end

function update_training_loss(report)
    Wandb.log(lg, Dict("metrics/training_loss" => report.training_losses[end]))
    return nothing
end

function update_epochs(epoch)
    Wandb.log(lg, Dict("log/epoch" => epoch)) 
    return nothing
end

import IterationControl: needs_loss, update!, done, takedown

struct WithModelLossDo{F<:Function}
    f::F
    stop_if_true::Bool
    stop_message::Union{String,Nothing}
end

# constructor:
WithModelLossDo(; f=(x,model)->@info("loss: $x"),
           stop_if_true=false,
           stop_message=nothing) = WithModelLossDo(f, stop_if_true, stop_message)
WithModelLossDo(f; kwargs...) = WithModelLossDo(; f=f, kwargs...)

IterationControl.needs_loss(::WithModelLossDo) = true

function IterationControl.update!(c::WithModelLossDo,
                 model,
                 verbosity,
                 n,
                 state=(loss=nothing, done=false))
    loss = IterationControl.loss(model)
    loss === nothing && throw(ERR_NEEDS_LOSS)
    r = c.f(loss, model)
    done = (c.stop_if_true && r isa Bool && r) ? true : false
    return (loss=loss, done=done)
end

IterationControl.done(c::WithModelLossDo, state) = state.done

function IterationControl.takedown(c::WithModelLossDo, verbosity, state)
    verbosity > 1 && @info "final loss: $(state.loss). "
    if state.done
        message = c.stop_message === nothing ?
            "Stop triggered by a `WithModelLossDo` control. " :
            c.stop_message
        verbosity > 0 && @info message
        return merge(state, (log = message,))
    else
        return merge(state, (log = "",))
    end
end


mutable struct SaveBest <: Function
    best_loss::Float64
    model_path::String
    threshold::Float64
end
function (callback::SaveBest)(loss, ic_model)
    update_loss(loss)
    mach = IterationControl.expose(ic_model)
    if loss < callback.best_loss
        @info "best model change" callback.best_loss loss
        callback.best_loss = loss
        model = mach.fitresult[1]
        model_state = Flux.state(model)
        jldsave(model_path; model_state=model_state, layers=layers, input_features=input_features)
    end
    if loss < callback.threshold
        return true
    end
    return false
end
