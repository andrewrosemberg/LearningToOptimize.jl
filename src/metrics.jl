function relative_rmse(ŷ, y)
    return sqrt(mean(((ŷ .- y) ./ y) .^ 2))
end

function relative_mae(ŷ, y)
    return mean(abs.((ŷ .- y) ./ y))
end
