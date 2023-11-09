import PowerModels: constraint_ohms_yt_from, constraint_ohms_yt_to, constraint_ohms_y_from, constraint_ohms_y_to

function _function_ohms_yt_from(branch::Dict)
    g, b = calc_branch_y(branch)
    tr, ti = calc_branch_t(branch)
    g_fr = branch["g_fr"]
    b_fr = branch["b_fr"]
    tm = branch["tap"]

    return  (vm_fr, vm_to, va_fr, va_to) ->  ((g+g_fr)/tm^2*vm_fr^2 + (-g*tr+b*ti)/tm^2*(vm_fr*vm_to*cos(va_fr-va_to)) + (-b*tr-g*ti)/tm^2*(vm_fr*vm_to*sin(va_fr-va_to)), # from
            -(b+b_fr)/tm^2*vm_fr^2 - (-b*tr-g*ti)/tm^2*(vm_fr*vm_to*cos(va_fr-va_to)) + (-g*tr+b*ti)/tm^2*(vm_fr*vm_to*sin(va_fr-va_to))) # to

end

function function_ohms_yt_from(branch::Dict)
    _function_ohms_yt_from(branch)
end

function PowerModels.constraint_ohms_yt_from(pm::AbstractACPModel, i::Int; nw::Int=nw_id_default)
    branch = ref(pm, nw, :branch, i)
    f_bus = branch["f_bus"]
    t_bus = branch["t_bus"]
    f_idx = (i, f_bus, t_bus)

    p_fr  = var(pm, nw,  :p, f_idx)
    q_fr  = var(pm, nw,  :q, f_idx)
    vm_fr = var(pm, nw, :vm, f_bus)
    vm_to = var(pm, nw, :vm, t_bus)
    va_fr = var(pm, nw, :va, f_bus)
    va_to = var(pm, nw, :va, t_bus)

    f_owms = function_ohms_yt_from(branch)
    f_owms_p, f_owms_q = f_owms(vm_fr, vm_to, va_fr, va_to)

    #constraint_ohms_yt_from(pm, nw, f_bus, t_bus, f_idx, t_idx, g, b, g_fr, b_fr, tr, ti, tm)
    JuMP.@constraint(pm.model, p_fr == f_owms_p) # @NL
    JuMP.@constraint(pm.model, q_fr == f_owms_q) # @NL
end

function _function_ohms_yt_to(branch::Dict)
    g, b = calc_branch_y(branch)
    tr, ti = calc_branch_t(branch)
    g_to = branch["g_to"]
    b_to = branch["b_to"]
    tm = branch["tap"]

    return  (vm_fr, vm_to, va_fr, va_to) ->  ((g+g_to)*vm_to^2 + (-g*tr-b*ti)/tm^2*(vm_to*vm_fr*cos(va_to-va_fr)) + (-b*tr+g*ti)/tm^2*(vm_to*vm_fr*sin(va_to-va_fr)), # from
            -(b+b_to)*vm_to^2 - (-b*tr+g*ti)/tm^2*(vm_to*vm_fr*cos(va_to-va_fr)) + (-g*tr-b*ti)/tm^2*(vm_to*vm_fr*sin(va_to-va_fr))) # to

end

function function_ohms_yt_to(branch::Dict)
    _function_ohms_yt_to(branch)
end

function PowerModels.constraint_ohms_yt_to(pm::AbstractACPModel, i::Int; nw::Int=nw_id_default)
    branch = ref(pm, nw, :branch, i)
    t_bus = branch["t_bus"]
    f_bus = branch["f_bus"]
    t_idx = (i, t_bus, f_bus)

    p_to  = var(pm, nw,  :p, t_idx)
    q_to  = var(pm, nw,  :q, t_idx)
    vm_fr = var(pm, nw, :vm, f_bus)
    vm_to = var(pm, nw, :vm, t_bus)
    va_fr = var(pm, nw, :va, f_bus)
    va_to = var(pm, nw, :va, t_bus)

    f_owms = function_ohms_yt_to(branch)
    f_owms_p, f_owms_q = f_owms(vm_fr, vm_to, va_fr, va_to)

    # constraint_ohms_yt_to(pm, nw, f_bus, t_bus, f_idx, t_idx, g, b, g_to, b_to, tr, ti, tm)
    JuMP.@constraint(pm.model, p_to == f_owms_p) # @NL
    JuMP.@constraint(pm.model, q_to == f_owms_q) # @NL
end

# function function_ohms_y_from(pm::AbstractACPModel, i::Int; nw::Int=nw_id_default)
#     branch = ref(pm, nw, :branch, i)
#     f_bus = branch["f_bus"]
#     t_bus = branch["t_bus"]
#     f_idx = (i, f_bus, t_bus)
#     t_idx = (i, t_bus, f_bus)

#     g, b = calc_branch_y(branch)
#     g_fr = branch["g_fr"]
#     b_fr = branch["b_fr"]
#     tm = branch["tap"]
#     ta = branch["shift"]

#     return  (vm_fr, vm_to, va_fr, va_to) ->  (g+g_fr)*(vm_fr/tm)^2 - g*vm_fr/tm*vm_to*cos(va_fr-va_to-ta) + -b*vm_fr/tm*vm_to*sin(va_fr-va_to-ta), # from
#             (vm_fr, vm_to, va_fr, va_to) -> -(b+b_fr)*(vm_fr/tm)^2 + b*vm_fr/tm*vm_to*cos(va_fr-va_to-ta) + -g*vm_fr/tm*vm_to*sin(va_fr-va_to-ta) # to

# end

# function constraint_ohms_y_from(pm::AbstractACPModel, i::Int; nw::Int=nw_id_default)
#     p_fr  = var(pm, nw,  :p, f_idx)
#     q_fr  = var(pm, nw,  :q, f_idx)
#     vm_fr = var(pm, nw, :vm, f_bus)
#     vm_to = var(pm, nw, :vm, t_bus)
#     va_fr = var(pm, nw, :va, f_bus)
#     va_to = var(pm, nw, :va, t_bus)

#     f_owms = function_ohms_y_from(pm, i; nw=nw)

#     # constraint_ohms_y_from(pm, nw, f_bus, t_bus, f_idx, t_idx, g, b, g_fr, b_fr, tm, ta)
#     JuMP.constraint(pm.model, p_fr == f_owms[1] # @NL(vm_fr, vm_to, va_fr, va_to))
#     JuMP.constraint(pm.model, q_fr == f_owms[2] # @NL(vm_fr, vm_to, va_fr, va_to))
# end

# function function_ohms_y_to(pm::AbstractACPModel, i::Int; nw::Int=nw_id_default)
#     branch = ref(pm, nw, :branch, i)
#     f_bus = branch["f_bus"]
#     t_bus = branch["t_bus"]
#     f_idx = (i, f_bus, t_bus)
#     t_idx = (i, t_bus, f_bus)

#     g, b = calc_branch_y(branch)
#     g_to = branch["g_to"]
#     b_to = branch["b_to"]
#     tm = branch["tap"]
#     ta = branch["shift"]

#     return  (vm_fr, vm_to, va_fr, va_to) ->  (g+g_to)*vm_to^2 - g*vm_to*vm_fr/tm*cos(va_to-va_fr+ta) + -b*vm_to*vm_fr/tm*sin(va_to-va_fr+ta), # from
#             (vm_fr, vm_to, va_fr, va_to) -> -(b+b_to)*vm_to^2 + b*vm_to*vm_fr/tm*cos(va_to-va_fr+ta) + -g*vm_to*vm_fr/tm*sin(va_to-va_fr+ta) # to

# end

# function constraint_ohms_y_to(pm::AbstractACPModel, i::Int; nw::Int=nw_id_default)
#     p_to  = var(pm, nw,  :p, t_idx)
#     q_to  = var(pm, nw,  :q, t_idx)
#     vm_fr = var(pm, nw, :vm, f_bus)
#     vm_to = var(pm, nw, :vm, t_bus)
#     va_fr = var(pm, nw, :va, f_bus)
#     va_to = var(pm, nw, :va, t_bus)

#     f_owms = function_ohms_y_to(pm, i; nw=nw)

#     # constraint_ohms_y_to(pm, nw, f_bus, t_bus, f_idx, t_idx, g, b, g_to, b_to, tm, ta)
#     JuMP.constraint(pm.model, p_to == f_owms[1] # @NL(vm_fr, vm_to, va_fr, va_to))
#     JuMP.constraint(pm.model, q_to == f_owms[2] # @NL(vm_fr, vm_to, va_fr, va_to))
# end
