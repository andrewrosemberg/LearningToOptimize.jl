

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


struct WithModelLossDo{F<:Function}
    f::F
    stop_if_true::Bool
    stop_message::Union{String,Nothing}
end

# constructor:
WithModelLossDo(; f=x->@info("loss: $x"),
           stop_if_true=false,
           stop_message=nothing) = WithModelLossDo(f, stop_if_true, stop_message)
WithModelLossDo(f; kwargs...) = WithModelLossDo(; f=f, kwargs...)

needs_loss(::WithModelLossDo) = true

function update!(c::WithModelLossDo,
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

done(c::WithModelLossDo, state) = state.done

function takedown(c::WithModelLossDo, verbosity, state)
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