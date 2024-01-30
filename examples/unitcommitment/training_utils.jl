
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