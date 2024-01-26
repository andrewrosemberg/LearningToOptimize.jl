##############
# Load Packages
##############

using LinearAlgebra
using Gurobi
using L2O
using JuMP
using Logging
using JuMP
using UnitCommitment
import ParametricOptInterface as POI
using DataFrames
using CSV
using UUIDs
using Arrow
using SparseArrays
using Statistics
using Random

import UnitCommitment:
    Formulation,
    KnuOstWat2018,
    MorLatRam2013,
    ShiftFactorsFormulation

function uc_load_disturbances!(rng, instance, nominal_loads; load_disturbances_range=-20:20)
    for i in 1:length(instance.buses)
        bus = instance.buses[i]
        bus.load = nominal_loads[i] .+ rand(rng, load_disturbances_range)
        bus.load = max.(bus.load, 0.0)
    end
end

"""
    Build model
"""
function build_model_uc(instance; solver=Gurobi.Optimizer, PoolSearchMode=2, PoolSolutions=100)
    # Construct model (using state-of-the-art defaults)
    model = UnitCommitment.build_model(
        instance = instance,
        optimizer = solver,
    )

    # Set solver attributes
    if !isnothing(PoolSearchMode)
        set_optimizer_attribute(model, "PoolSearchMode", PoolSearchMode)
        set_optimizer_attribute(model, "PoolSolutions", PoolSolutions)
    end

    return model
end


function tuple_2_name(smb)
    str = string(smb[1])
    for i in 2:length(smb)
        str = str * "_" * string(smb[i])
    end
    return str
end

function bin_variables_retriever(model)
    bin_vars = vcat(
        collect(values(model[:is_on])), 
        collect(values(model[:startup])), 
        collect(values(model[:switch_on])),
        collect(values(model[:switch_off]))
    )

    bin_vars_names = vcat(
        "is_on_" .* tuple_2_name.(collect(keys(model[:is_on]))), 
        "startup_" .* tuple_2_name.(collect(keys(model[:startup]))), 
        "switch_on_" .* tuple_2_name.(collect(keys(model[:switch_on]))),
        "switch_off_" .* tuple_2_name.(collect(keys(model[:switch_off])))
    )
    return bin_vars, bin_vars_names
end

mutable struct StorageCallback <: Function
    model
    bin_vars
    my_storage_vars
    my_storage_obj
    is_relaxed
end
function StorageCallback(model, bin_vars; maxiter=999999)
    return StorageCallback(
        model,
        bin_vars,
        [],
        [],
        [],
    )
end

function (callback::StorageCallback)(cb_data, cb_where::Cint)
    # You can select where the callback is run
    if cb_where == GRB_CB_MIPNODE
        # prep
        obj_terms = objective_function(callback.model).terms
        obj_terms_gurobi = [obj_terms[var] for var in all_variables(callback.model) if haskey(obj_terms, var)]
        num_all_var = num_variables(callback.model)
        # save
        resultobj = Ref{Cint}()
        GRBcbget(cb_data, cb_where, GRB_CB_MIPNODE_STATUS, resultobj)
        if resultobj[] != GRB_OPTIMAL
            return  # Solution is something other than optimal.
        end
        gurobi_indexes_all = [Gurobi.column(backend(callback.model).optimizer.model, callback.model.moi_backend.model_to_optimizer_map[var.index]) for var in all_variables(callback.model) if haskey(obj_terms, var)]
        gurobi_indexes_bin = [Gurobi.column(backend(callback.model).optimizer.model, callback.model.moi_backend.model_to_optimizer_map[callback.bin_vars[i].index]) for i in 1:length(callback.bin_vars)]
        
        resultP = Vector{Cdouble}(undef, num_all_var)
        GRBcbget(cb_data, cb_where, GRB_CB_MIPNODE_REL, resultP)
        push!(callback.my_storage_vars, resultP[gurobi_indexes_bin])
        # Get the objective value
        push!(callback.my_storage_obj, dot(obj_terms_gurobi, resultP[gurobi_indexes_all]))
        # mark as relaxed
        push!(callback.is_relaxed, 1)
        return
    end
    if cb_where == GRB_CB_MIPSOL
        # Before querying `callback_value`, you must call:
        Gurobi.load_callback_variable_primal(cb_data, cb_where)
        # Get the values of the variables
        x = [callback_value(cb_data, var) for var in callback.bin_vars]
        push!(callback.my_storage_vars, x)
        # Get the objective value
        obj = Ref{Cdouble}()
        GRBcbget(cb_data, cb_where, GRB_CB_MIPSOL_OBJ, obj)
        push!(callback.my_storage_obj, obj[])
        # mark as not relaxed
        push!(callback.is_relaxed, 0)
        return
    end
end

"""
    Build branch and bound dataset
"""
function uc_bnb_dataset(instance, save_file; model=build_model_uc(instance), data_dir=pwd(), batch_id = uuid1(), filetype=ArrowFile)
    ##############
    # Solve and store solutions
    ##############

    bin_vars, bin_vars_names = bin_variables_retriever(model)

    @assert all([is_binary(var) for var in bin_vars])
    @assert length(bin_vars) == length(all_binary_variables(model))

    # Callback
    my_callback_function = StorageCallback(model, bin_vars)
    # Set callback
    MOI.set(model, Gurobi.CallbackFunction(), my_callback_function)

    # JuMP.optimize!(model)
    UnitCommitment.optimize!(model)
    @info "status: " termination_status(model)
    # push optimal solution
    x = [value(var) for var in bin_vars]
    push!(my_callback_function.my_storage_vars, x)
    optimal_obj = objective_value(model)
    push!(my_callback_function.my_storage_obj, optimal_obj)
    # mark as not relaxed
    push!(my_callback_function.is_relaxed, 0)

    @info "Solved nominal instance" optimal_obj length(my_callback_function.my_storage_vars)

    # post process
    is_relaxed = findall(x -> x == 1, my_callback_function.is_relaxed)
    my_storage_vars = my_callback_function.my_storage_vars
    my_storage_obj = my_callback_function.my_storage_obj

    # Data
    X = hcat(my_storage_vars...)'[:,:]
    y = convert.(Float64, my_storage_obj[:,:])

    # Save solutions
    # Input
    instances_ids = [uuid1() for i in 1:length(my_storage_vars)]
    df_in = DataFrame(X, Symbol.(bin_vars_names))
    df_in.id = instances_ids
    # Save Loads
    for bus in instance.buses
        for t in 1:instance.time
            df_in[!, Symbol("load_" * string(bus.name) * "_" * string(t))] = fill(bus.load[t], length(instances_ids))
        end
    end
    # Output
    df_out = DataFrame(Dict(:objective => y[:,1]))
    df_out.id = instances_ids
    df_out.time = fill(solve_time(model), length(instances_ids))
    df_out.status = fill("LOCALLY_SOLVED", length(instances_ids))
    df_out.primal_status = fill("FEASIBLE_POINT", length(instances_ids))
    df_out.dual_status = fill("FEASIBLE_POINT", length(instances_ids))
    # mark as optimal
    df_out.status[end] = "OPTIMAL"
    # mark as relaxed
    df_out.status[is_relaxed] = fill("INFEASIBLE", length(is_relaxed))
    df_out.primal_status[is_relaxed] = fill("INFEASIBLE_POINT", length(is_relaxed))
    df_out.dual_status[is_relaxed] = fill("INFEASIBLE_POINT", length(is_relaxed))

    # Save
    if filetype === ArrowFile
        Arrow.write(joinpath(data_dir, save_file * "_input_" * string(batch_id) * ".arrow"), df_in)
        Arrow.write(joinpath(data_dir, save_file * "_output_" * string(batch_id) * ".arrow"), df_out)
    else
        CSV.write(joinpath(data_dir, save_file * "_input_" * string(batch_id) * ".csv"), df_in)
        CSV.write(joinpath(data_dir, save_file * "_output_" * string(batch_id) * ".csv"), df_out)
    end

    @info "Saved dataset to $(data_dir)" batch_id length(instances_ids) length(is_relaxed) optimal_obj

    return
end

"""
    Enhance dataset
"""
function uc_random_dataset!(instance, save_file; model=build_model_uc(instance), delete_objective=false, inner_solver=() -> POI.Optimizer(Gurobi.Optimizer()), data_dir=pwd(), filetype=ArrowFile, num_s = 1000, non_zero_units = 0.15, batch_id = uuid1())
    MOI.set(model, Gurobi.CallbackFunction(), nothing)
    bin_vars, bin_vars_names = bin_variables_retriever(model)
    # Remove binary constraints
    upper_model, inner_2_upper_map, cons_mapping = copy_binary_model(model)

    # delete binary constraints from inner model
    delete_binary_terms!(model; delete_objective=delete_objective)
    # add deficit constraints
    add_deficit_constraints!(model)
    # link binary variables from upper to inner model
    upper_2_inner = fix_binary_variables!(model, inner_2_upper_map)
    # get parameters from inner model in the right order
    u_inner = [upper_2_inner[inner_2_upper_map[var]] for var in bin_vars]
    # set names
    set_name.(u_inner, bin_vars_names)
    # set solver
    set_optimizer(model, inner_solver)
    # Parameter values
    u_values = abs.(Matrix(hcat([sprandn(length(u_inner), 1, non_zero_units) for i in 1:num_s]...)'))
    u_values = min.(u_values, 1.0)
    units_on = sum(u_values, dims=1)
    @info "Number of units on: " mean(units_on) std(units_on) minimum(units_on) maximum(units_on)
    parameter_values = Dict(u_inner .=> Array.(eachcol(u_values)))
    # The iterator
    problem_iterator = ProblemIterator(parameter_values)
    input_file = save_file * "_input_" * string(batch_id)
    save(problem_iterator, joinpath(data_dir, input_file), filetype)
    input_file = input_file * "." * string(filetype)
    # Add load data to input file
    df_in = if filetype === ArrowFile
        Arrow.Table(joinpath(data_dir, input_file)) |> DataFrame
    else
        CSV.read(joinpath(data_dir, input_file), DataFrame)
    end
    # delete old file
    rm(joinpath(data_dir, input_file))
    for bus in instance.buses
        for t in 1:instance.time
            df_in[!, Symbol("load_" * string(bus.name) * "_" * string(t))] = fill(bus.load[t], length(df_in.id))
        end
    end
    if filetype === ArrowFile
        Arrow.write(joinpath(data_dir, input_file), df_in)
    else
        CSV.write(joinpath(data_dir, input_file), df_in)
    end
    # CSV recorder to save the optimal primal and dual decision values
    output_file = save_file * "_output_" * string(batch_id)
    recorder = Recorder{filetype}(joinpath(data_dir, output_file); model=model)
    output_file = output_file * "." * string(filetype)

    # Finally solve all problems described by the iterator
    solve_batch(problem_iterator, recorder)
end