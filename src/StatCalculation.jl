using ProgressMeter
using Random
using StaticArrays
using LinearAlgebra
using Polyester

# ==============================================================================
# --- Central Plugin Registry (Dimension-Based) ---
# ==============================================================================

# The active, mutable global registry template (starts completely empty)
const _ACTIVE_STAT_REGISTRY = Ref{Dict{Symbol, Union{Symbol, Vector{Symbol}}}}(Dict())

"""
    get_stat_registry()

Retrieves the currently active global statistic registry template.
"""
get_stat_registry() = _ACTIVE_STAT_REGISTRY[]

"""
    set_stat_registry!(new_registry::Dict{Symbol, Union{Symbol, Vector{Symbol}}})

Completely overwrites the active global statistic registry with a new dictionary. 
"""
function set_stat_registry!(new_registry::Dict{Symbol, Union{Symbol, Vector{Symbol}}})
    _ACTIVE_STAT_REGISTRY[] = deepcopy(new_registry)
    @info "Global statistic registry has been overwritten."
end

"""
    reset_stat_registry!()

Clears the global statistic registry entirely.
"""
function reset_stat_registry!()
    empty!(_ACTIVE_STAT_REGISTRY[])
    @info "Global statistic registry has been cleared."
end

"""
    set_stat_preset!(name::String)

Dynamically loads a predefined set of statistics and their `calc_stat` overloads from the `src/presets/` directory.

# Examples
```julia-repl
julia> set_stat_preset!("hyperbolic")
```
"""
function set_stat_preset!(name::String)
    preset_path = joinpath(@__DIR__, "../presets", "$name.jl")
    if isfile(preset_path)
        # Evaluates the preset file inside the PDEStudioCore module namespace
        Base.include(@__MODULE__, preset_path)
        @info "Successfully loaded stat preset: $name"
    else
        @error "Preset file not found: $preset_path"
    end
end

"""
    add_stat!(sim_data::AbstractSimData, name::Symbol, value, kept_dims::Union{Symbol, Vector{Symbol}})

Appends a fully custom, pre-calculated statistic directly to a simulation dataset. 
Registers the dimensions it keeps so the Plotter UI knows exactly how to slice and display it.

# Arguments
- `sim_data::AbstractSimData`: The Eulerian or Lagrangian data to attach the statistic to.
- `name::Symbol`: The identifier for the custom statistic.
- `value`: The computed statistical array or scalar.
- `kept_dims::Union{Symbol, Vector{Symbol}}`: The dimensions this array spans.
"""
function add_stat!(sim_data::AbstractSimData{D, DS, M, T}, name::Symbol, value, kept_dims::Union{Symbol, Vector{Symbol}}) where {D, DS, M, T}
    value_vec = value isa Real ? SVector{M, T}([T(value) for _ in 1:M]) : value
    sim_data.stats[name] = value_vec
    sim_data.domain.stat_registry[name] = kept_dims
    @info "Added custom stat '$name' keeping dimensions: $kept_dims"
end

# Ultimate Fallback (Safely returns an SVector of NaNs matching the component count)
"""
    calc_stat(stat, fixed_coords, u, ana, domain)

The ultimate fallback dispatcher for statistical calculations. 
If a custom statistic is evaluated without a specific typed overload, this safely returns an `SVector` of `NaN`s matching the field component count.
"""
calc_stat(stat, fixed_coords, u, ana, domain) = zero(eltype(u)) .* NaN

"""
    register_stat!(name::Symbol, kept_dims::Symbol)

Registers a custom statistic into the global registry and defines which dimensions should be retained during integration.

# Arguments
- `name::Symbol`: The unique identifier for your custom statistic.
- `kept_dims::Symbol`: The dimensions to retain. Valid aliases are `:all`, `:space`, or `:time`.

# Examples
```julia-repl
julia> register_stat!(:l2_error, :time)
```
"""
function register_stat!(name::Symbol, kept_dims::Union{Symbol, Vector{Symbol}})
    _ACTIVE_STAT_REGISTRY[][name] = kept_dims
    @info "Registered statistic :$name keeping dimensions: $kept_dims"
end

"""
    get_kept_dims(stat::Symbol, domain::DomainInfo)

Translates generic aliases (:all, :space, :time) into exact dimension symbols 
using the domain's local registry and specific time_dim.
"""
function get_kept_dims(stat::Symbol, domain::DomainInfo)
    reg_val = get(domain.stat_registry, stat, Symbol[])
    
    if reg_val === :all
        return collect(domain.dim_keys)
    elseif reg_val === :space
        return filter(d -> d !== domain.time_dim, collect(domain.dim_keys))
    elseif reg_val === :time
        return isnothing(domain.time_dim) ? Symbol[] : [domain.time_dim]
    elseif reg_val isa Vector{Symbol}
        return reg_val
    else
        return Symbol[]
    end
end

"""
    get_kept_indices(stat::Symbol, domain::DomainInfo)

Maps the retained dimension symbols for a given statistic to their corresponding numerical indices within the domain's coordinate axes.

# Arguments
- `stat::Symbol`: The registered name of the statistic.
- `domain::DomainInfo`: The bounding box metadata of the simulation run.

# Returns
- `Vector{Int}`: A sorted list of dimension indices.
"""
function get_kept_indices(stat::Symbol, domain::DomainInfo)
    kept_dims = get_kept_dims(stat, domain)
    
    indices = Int[]
    for dim in kept_dims
        idx = findfirst(==(dim), domain.dim_keys)
        if !isnothing(idx)
            push!(indices, idx)
        end
    end
    
    return sort(indices)
end

"""
    get_integration_measure(stat::Symbol, domain::DomainInfo{D}) where {D}

Computes the combined scalar volumetric integration measure (e.g., dx * dy * dt) for the dimensions that are being integrated out for a specific statistic. 
Dimensions that are retained do not contribute to this measure.

# Arguments
- `stat::Symbol`: The registered name of the statistic.
- `domain::DomainInfo`: The domain containing the uniform spacing metadata.

# Returns
- A floating-point scalar representing the combined spacing measure.
"""
function get_integration_measure(stat::Symbol, domain::DomainInfo{D}) where {D}
    kept_dims = get_kept_dims(stat, domain)
    measure = 1.0
    for d in 1:D
        if !(domain.dim_keys[d] in kept_dims)
            measure *= domain.spacing[d]
        end
    end
    return measure
end

# ==============================================================================
# --- MAIN PIPELINE (Entry Points) ---
# ==============================================================================

"""
    remove_nan_stats!(stats_dict::StatDict)

Iterates over a statistical dictionary and removes any registered statistic where 
every value evaluates to `NaN`. This typically occurs when a statistic requiring an 
analytical reference solution is evaluated without one provided.

# Arguments
- `stats_dict::StatDict`: The dictionary of computed statistics attached to a simulation object.
"""
function remove_nan_stats!(stats_dict::StatDict)
    keys_to_remove = Symbol[]
    for (name, val) in stats_dict
        if is_all_nan(val)
            @info "Removing statistic :$name because all values are NaN (no reference data)."
            push!(keys_to_remove, name)
        end
    end
    
    for k in keys_to_remove; delete!(stats_dict, k); end
end
"""
    is_all_nan(val)

A recursive helper function to accurately detect if an entire array structure evaluates to `NaN`. 
Safely handles Eulerian multidimensional arrays, flat scalar `SVector`s, and nested Lagrangian 
particle series.

# Arguments
- `val`: The array or vector to evaluate.

# Returns
- `Bool`: `true` if every element in the nested structure is `NaN`.
"""
function is_all_nan(val)
    # Handle Nested Lagrangian Fields
    if val isa Vector{<:Vector} 
        return all(vec -> all(svec -> any(isnan, svec), vec), val)
    # Handle Eulerian Tensors & Lagrangian Series
    elseif val isa AbstractArray 
        return all(svec -> any(isnan, svec), val)
    # Handle Base Scalars
    elseif val isa SVector 
        return any(isnan, val)
    else
        return false
    end
end

"""
    calculate_all_stats!(sim_data::AbstractSimData, ref_func; force_overwrite = false, kwargs...)

The core statistical integration engine. Evaluates all metrics currently registered in the 
simulation's `stat_registry` across the provided mathematical dataset. 

If a `ref_func` (analytical truth) is provided, it generates a pointwise perfect cache mapping 
over the exact spatial layout of the data before executing multithreaded reductions. 

# Arguments
- `sim_data::AbstractSimData`: The loaded simulation object to analyze.
- `ref_func::Function`: The continuous mathematical reference function (can be `nothing`).

# Keyword Arguments
- `force_overwrite::Bool`: Re-evaluates and overwrites metrics already present in `sim_data.stats`. Default is `false`.
"""
function calculate_all_stats!(sim_data::AbstractSimData, ref_func; force_overwrite = false, kwargs...)
    # 1. Generate full analytical field upfront (NaNs or exact)
    u_ana = isnothing(ref_func) ? generate_pointwise_nan(sim_data) : generate_pointwise_reference(sim_data, ref_func)
    
    # 2. Process all registered statistics dynamically
    stat_change = false
    for (stat_name, kept_dims) in sim_data.domain.stat_registry
        if stat_name == :Solution; continue end
        if haskey(sim_data.stats, stat_name) && !force_overwrite; continue end
        
        res = _calc_stat!(sim_data, u_ana, stat_name)
        
        if !isnothing(res)
            if !is_all_nan(res)
                sim_data.stats[stat_name] = res
                stat_change = true
            else
                if haskey(sim_data.stats, stat_name)
                    delete!(sim_data.stats, stat_name)
                    stat_change = true
                end
            end
        end
    end
    
    # 3. Cleanup and Save
    remove_nan_stats!(sim_data.stats)
    save_sim_data(sim_data; overwrite=stat_change)
end

# ==============================================================================
# --- EULERIAN STATISTICAL REDUCTIONS ---
# ==============================================================================

# 1. The inline helper that safely executes the lambda out of @batch's sight
@inline function _extract_coords(axes, kept_idx, I, ::Val{L}, ::Type{T}) where {L, T}
    return SVector{L, T}(ntuple(d -> axes[kept_idx[d]][I[d]], Val(L)))
end

# 2. The type-stable barrier function containing the clean loop
function _batch_stat_calc!(res, u_slices, ana_slices, axes, kept_idx, domain, stat_name, ::Val{L}, ::Type{T}) where {L, T}
    @batch for i in eachindex(u_slices)
        I = CartesianIndices(u_slices)[i]
        
        # Zero-allocation SVector generation!
        fixed_coords = _extract_coords(axes, kept_idx, I, Val(L), T)
        
        res[i] = calc_stat(Val(stat_name), fixed_coords, vec(u_slices[i]), vec(ana_slices[i]), domain)
    end
end

# 3. The main entry point
"""
    _calc_stat!(sim_data::ESimData, u_ana, stat_name::Symbol)
    _calc_stat!(sim_data::LSimData, u_ana, stat_name::Symbol)

Internal backend dispatch for executing statistical reductions. It slices the dense Eulerian tensor 
or Lagrangian series along the correct retained dimensions, and uses Polyester `@batch` multithreading 
to integrate out the ignored axes efficiently.

# Arguments
- `sim_data`: The Eulerian or Lagrangian data.
- `u_ana`: The pre-calculated pointwise analytical array (or array of NaNs).
- `stat_name::Symbol`: The registered name of the statistic to calculate.

# Returns
- The integrated statistical array mapping exactly to the retained dimensions.
"""
function _calc_stat!(sim_data::ESimData{D, DS, M, T}, u_ana, stat_name::Symbol) where {D, DS, M, T}
    kept_idx = get_kept_indices(stat_name, sim_data.domain)
    
    if isempty(kept_idx)
        return calc_stat(Val(stat_name), SVector{0, T}(), vec(sim_data.u), vec(u_ana), sim_data.domain)
    end
    
    L = length(kept_idx)
    out_sz = ntuple(d -> length(sim_data.axes[kept_idx[d]]), L)
    res = Array{SVector{M, T}, L}(undef, out_sz...)
    
    u_slices = eachslice(sim_data.u, dims=Tuple(kept_idx))
    ana_slices = eachslice(u_ana, dims=Tuple(kept_idx))
    
    # Cross the function barrier to turn runtime `L` into compile-time `Val(L)`
    _batch_stat_calc!(res, u_slices, ana_slices, sim_data.axes, kept_idx, sim_data.domain, stat_name, Val(L), T)
    
    return res
end

# ==============================================================================
# --- LAGRANGIAN STATISTICAL REDUCTIONS ---
# ==============================================================================

function _calc_stat!(sim_data::LSimData{D, DS, M, T}, u_ana, stat_name::Symbol) where {D, DS, M, T}
    kept_dims = get_kept_dims(stat_name, sim_data.domain)
    
    is_series = kept_dims == [sim_data.domain.time_dim] || (D == DS && isempty(kept_dims))
    is_field = length(kept_dims) == D
    Nt = length(sim_data.t)
    
    if is_series
        res = Vector{SVector{M, T}}(undef, Nt)
        @batch for t_idx in 1:Nt
            fixed = D > DS ? SVector{1, T}(sim_data.t[t_idx]) : SVector{0, T}()
            res[t_idx] = calc_stat(Val(stat_name), fixed, sim_data.u[t_idx], u_ana[t_idx], sim_data.domain)
        end
        return res
        
    elseif is_field
        res = Vector{Vector{SVector{M, T}}}(undef, Nt)
        @batch for t_idx in 1:Nt
            Np = length(sim_data.x[t_idx])
            res_t = Vector{SVector{M, T}}(undef, Np)
            for p_idx in 1:Np
                fixed = D > DS ? SVector{D, T}(sim_data.x[t_idx][p_idx]..., sim_data.t[t_idx]) : SVector{D, T}(sim_data.x[t_idx][p_idx]...)
                res_t[p_idx] = calc_stat(Val(stat_name), fixed, (sim_data.u[t_idx][p_idx],), (u_ana[t_idx][p_idx],), sim_data.domain)
            end
            res[t_idx] = res_t
        end
        return res
    end
end

# ==============================================================================
# --- ANALYTICAL CACHE GENERATOR ---
# ==============================================================================
"""
    generate_pointwise_nan(data::LSimData)
    generate_pointwise_nan(data::ESimData)

Generates an exact structural replica of the simulation's field array (`u`), filled entirely 
with `NaN`s. This allows the statistical pipeline to maintain type stability and run seamlessly 
even when an analytical reference function is missing.

# Arguments
- `data`: The Eulerian or Lagrangian simulation data.

# Returns
- An array mirroring `data.u` containing `SVector{M, T}(NaN, NaN...)`.
"""
function generate_pointwise_nan(data::LSimData{D, DS, M, T}) where {D, DS, M, T}
    return [fill(SVector{M, T}(ntuple(_ -> T(NaN), M)), length(x)) for x in data.x]
end

function generate_pointwise_nan(data::ESimData{D, DS, M, T}) where {D, DS, M, T}
    return fill(SVector{M, T}(ntuple(_ -> T(NaN), M)), size(data.u))
end

"""
    generate_pointwise_reference(data::LSimData, ref_func)
    generate_pointwise_reference(data::ESimData, ref_func)

Generates a precise pointwise cache of the exact analytical truth by evaluating the continuous 
`ref_func` at the exact spatial and temporal coordinates of every grid node or Lagrangian particle.
Uses Polyester `@batch` for highly efficient multithreaded evaluation.

# Arguments
- `data`: The simulation data providing the coordinate targets.
- `ref_func::Function`: The continuous mathematical function returning an `SVector{M, T}`.

# Returns
- An array mirroring `data.u` populated with the exact analytical values.
"""
function generate_pointwise_reference(ldata::LSimData{D, DS, M, T}, ref_func) where {D, DS, M, T}
    Nt = length(ldata.t)
    u_ana = Vector{Vector{SVector{M, T}}}(undef, Nt)
    _ref = ref_func 
    
    @batch for t_idx in 1:Nt
        t_val = ldata.t[t_idx]
        xs = ldata.x[t_idx]
        if D > DS
            u_ana[t_idx] = [_ref(SVector{D, T}(x..., t_val)) for x in xs]
        else
            u_ana[t_idx] = [_ref(SVector{D, T}(x...)) for x in xs]
        end
    end
    return u_ana
end

function generate_pointwise_reference(edata::ESimData{D, DS, M, T}, ref_func) where {D, DS, M, T}
    u_ana = similar(edata.u)
    _ref = ref_func
    
    @batch for i in eachindex(edata.u)
        I = CartesianIndices(edata.u)[i]
        st = SVector{D, T}(ntuple(d -> edata.axes[d][I[d]], Val(D)))
        u_ana[i] = _ref(st)
    end
    return u_ana
end