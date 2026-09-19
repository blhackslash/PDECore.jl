"""
    _get_lsim_bounds(x::Vector{Vector{SVector{DS, T}}}) where {DS, T}

An internal helper function that scans the entire Lagrangian particle history to determine the absolute minimum and maximum spatial coordinates across all time steps.

# Arguments
- `x::Vector{Vector{SVector{DS, T}}}`: The nested vectors containing particle coordinates over time.

# Returns
- A tuple of `(mins, maxs)`, where each is a `Tuple` of bounds for the spatial dimensions.
"""
function _get_lsim_bounds(x::Vector{Vector{SVector{DS, T}}}) where {DS, T}
    mins, maxs = fill(T(Inf), DS), fill(T(-Inf), DS)
    for step in x; for p in step; for d in 1:DS
        mins[d], maxs[d] = min(mins[d], p[d]), max(maxs[d], p[d])
    end; end; end
    return Tuple(mins), Tuple(maxs)
end

# ==============================================================================
# --- EULERIAN CONSTRUCTORS (Spacetime Tensors) ---
# ==============================================================================

# 1D Space + 1D Time = 2D Spacetime Tensor
"""
    create_sim_data(x, u, t, params; kwargs...)

Constructs an Eulerian simulation dataset for a 1D spatial domain and 1D temporal domain (yielding a 2D spacetime tensor). 
Automatically calculates uniform grid spacing, initializes a bounding `DomainInfo`, and registers the primary solution in the stats dictionary.

# Arguments
- `x::AbstractVector{<:Real}`: The 1D spatial coordinate vector.
- `u::AbstractMatrix{SVector{M, T}}`: The 2D spacetime field data.
- `t::AbstractVector{<:Real}`: The discrete time vector.
- `params::ParamDict`: The parameters used to generate this data.

# Keyword Arguments
- `xmins`, `xmaxs`, `tmin`, `tmax`: Manual overrides for the domain bounding box.
- `time_dim::Union{Nothing,Symbol}`: The identifier for the temporal axis (defaults to `:t`).
- `x_dim::Symbol`: The identifier for the spatial axis (defaults to `:x`).

# Returns
- `ESimData`: The strictly typed Eulerian dataset.
"""
function create_sim_data(
    x::AbstractVector{<:Real}, u::AbstractMatrix{SVector{M, T}}, t::AbstractVector{<:Real}, params::ParamDict;
    xmins=nothing, xmaxs=nothing, tmin=nothing, tmax=nothing, time_dim::Union{Nothing,Symbol}=:t, x_dim::Symbol=:x
) where {M, T<:Real}
    DS, D = 1, 2
    
    _mins = SVector{D, T}(
        isnothing(xmins) ? T(minimum(x)) : T(xmins[1]), 
        isnothing(tmin) ? T(minimum(t)) : T(tmin)
    )
    _maxs = SVector{D, T}(
        isnothing(xmaxs) ? T(maximum(x)) : T(xmaxs[1]), 
        isnothing(tmax) ? T(maximum(t)) : T(tmax)
    )
    
    axes = (Vector{T}(x), Vector{T}(t))
    spacing = SVector{D, T}(
        length(x) > 1 ? (_maxs[1] - _mins[1]) / T(length(x) - 1) : one(T),
        length(t) > 1 ? (_maxs[2] - _mins[2]) / T(length(t) - 1) : one(T)
    )
    
    registry = deepcopy(_ACTIVE_STAT_REGISTRY[])
    registry[:Solution] = :all
    
    dim_keys = (x_dim, time_dim)
    domain = DomainInfo{D, T}(dim_keys, _mins, _maxs, spacing, time_dim, registry)

    u_typed = u isa AbstractMatrix{SVector{M, T}} ? u : [SVector{M, T}(v) for v in u]
    stats_dict = StatDict{M, T}(:Solution => u_typed)

    return ESimData{D, DS, M, T}(params, domain, axes, u_typed, stats_dict)
end

# 2D Space + 1D Time = 3D Spacetime Tensor
"""
    create_sim_data(x_grid, y_grid, u, t, params; kwargs...)

Constructs an Eulerian simulation dataset for a 2D spatial domain and 1D temporal domain (yielding a 3D spacetime tensor). Extracts the base 1D axes from the provided 2D grid matrices.

# Arguments
- `x_grid::AbstractMatrix{<:Real}`: The 2D matrix of x-coordinates.
- `y_grid::AbstractMatrix{<:Real}`: The 2D matrix of y-coordinates.
- `u::AbstractArray{SVector{M, T}, 3}`: The 3D spacetime field data.
- `t::AbstractVector{<:Real}`: The discrete time vector.
- `params::ParamDict`: The parameters used to generate this data.

# Returns
- `ESimData`: The strictly typed Eulerian dataset representing 2D spatial fields over time.
"""
function create_sim_data(
    x_grid::AbstractMatrix{<:Real}, y_grid::AbstractMatrix{<:Real}, u::AbstractArray{SVector{M, T}, 3}, t::AbstractVector{<:Real}, params::ParamDict;
    xmins=nothing, xmaxs=nothing, tmin=nothing, tmax=nothing, time_dim::Union{Nothing,Symbol}=:t, x_dim::Symbol=:x,y_dim::Symbol=:y
) where {M, T<:Real}
    DS, D = 2, 3
    
    x_axis, y_axis = vec(x_grid[:, 1]), vec(y_grid[1, :])
    
    _mins = SVector{D, T}(
        isnothing(xmins) ? T(minimum(x_axis)) : T(xmins[1]), 
        isnothing(xmins) ? T(minimum(y_axis)) : T(xmins[2]), 
        isnothing(tmin) ? T(minimum(t)) : T(tmin)
    )
    _maxs = SVector{D, T}(
        isnothing(xmaxs) ? T(maximum(x_axis)) : T(xmaxs[1]), 
        isnothing(xmaxs) ? T(maximum(y_axis)) : T(xmaxs[2]), 
        isnothing(tmax) ? T(maximum(t)) : T(tmax)
    )
    
    axes = (Vector{T}(x_axis), Vector{T}(y_axis), Vector{T}(t))
    spacing = SVector{D, T}(
        length(x_axis) > 1 ? (_maxs[1] - _mins[1]) / T(length(x_axis) - 1) : one(T),
        length(y_axis) > 1 ? (_maxs[2] - _mins[2]) / T(length(y_axis) - 1) : one(T),
        length(t) > 1 ? (_maxs[3] - _mins[3]) / T(length(t) - 1) : one(T)
    )
    
    registry = deepcopy(_ACTIVE_STAT_REGISTRY[])
    registry[:Solution] = :all
    
    dim_keys = (x_dim, y_dim, time_dim)
    domain = DomainInfo{D, T}(dim_keys, _mins, _maxs, spacing, time_dim, registry)

    u_typed = u isa AbstractArray{SVector{M, T}, 3} ? u : [SVector{M, T}(v) for v in u]
    stats_dict = StatDict{M, T}(:Solution => u_typed)

    return ESimData{D, DS, M, T}(params, domain, axes, u_typed, stats_dict)
end

# ==============================================================================
# --- LAGRANGIAN CONSTRUCTORS ---
# ==============================================================================
"""
    create_sim_data(x, u, t, params; kwargs...)

Constructs a Lagrangian simulation dataset. If spatial bounds are not explicitly provided, it automatically determines them by scanning the particle coordinate vectors. It also computes an average equivalent spatial resolution (`avg_dx`) based on the particle count and domain volume.

# Arguments
- `x::Vector{Vector{SVector{DS, T}}}`: Nested vectors of particle spatial coordinates per time step.
- `u::Vector{Vector{SVector{M, T}}}`: Nested vectors of particle state vectors per time step.
- `t::Vector{T}`: The discrete time vector.
- `params::ParamDict`: The configuration parameters.

# Returns
- `LSimData`: The strictly typed Lagrangian dataset.
"""
function create_sim_data(
    x::Vector{Vector{SVector{DS, T}}}, u::Vector{Vector{SVector{M, T}}}, t::Vector{T}, params::ParamDict;
    xmins=nothing, xmaxs=nothing, tmin=nothing, tmax=nothing, time_dim::Union{Nothing,Symbol}=:t
) where {DS, M, T<:Real}
    D = DS + 1 
    
    auto_mins, auto_maxs = _get_lsim_bounds(x)
    _xmins = isnothing(xmins) ? auto_mins : Tuple(T.(xmins))
    _xmaxs = isnothing(xmaxs) ? auto_maxs : Tuple(T.(xmaxs))
    _tmin  = isnothing(tmin)  ? T(minimum(t)) : T(tmin)
    _tmax  = isnothing(tmax)  ? T(maximum(t)) : T(tmax)
    
    N_p = max(1, maximum(length.(x)))
    avg_dx = T((prod(_xmaxs .- _xmins) / N_p)^(1/DS))
    dt = length(t) > 1 ? (_tmax - _tmin) / T(length(t) - 1) : one(T)
    
    dim_keys = ((:x, :y, :z)[1:DS]..., time_dim)
    
    mins = SVector{D, T}(_xmins..., _tmin)
    maxs = SVector{D, T}(_xmaxs..., _tmax)
    spacing = SVector{D, T}(ntuple(d -> avg_dx, Val(DS))..., dt)
    
    registry = deepcopy(_ACTIVE_STAT_REGISTRY[])
    registry[:Solution] = :all
    domain = DomainInfo{D, T}(dim_keys, mins, maxs, spacing, time_dim, registry)
    stats_dict = StatDict{M, T}(:Solution => u)

    return LSimData{D, DS, M, T}(params, domain, t, x, u, stats_dict)
end


"""
    get_time_dim(domain::DomainInfo)

Retrieves the numerical index of the temporal dimension within the domain's sequence of dimension keys.

# Arguments
- `domain::DomainInfo`: The bounding box metadata of the simulation run.

# Returns
- An integer index, or `nothing` if the dataset represents static, purely spatial data.
"""
get_time_dim(domain::DomainInfo) = findfirst(==(domain.time_dim), domain.dim_keys)

# ==============================================================================
# --- CONVERSIONS ---
# ==============================================================================
# --- ALGORITHM: DYNAMIC SCATTER ---
"""
    interpolate_to_grid!(::Val{:scatter}, u_euler, w_euler, e_fields_tup, ldata, field_vals_tup, nan_vec, s_mins, s_maxs, s_dx, s_inv_dx, grid_shape)

An internal multithreaded algorithm that scatters Lagrangian particles and their associated statistical fields onto a structured Eulerian grid. 
It utilizes a distance-based weighting mechanism defined by a calculated effective smoothing radius, preventing zero-division via clamped minimum distances.

# Arguments
- `u_euler`: The preallocated target grid for the primary solution.
- `w_euler`: The preallocated weight accumulator grid.
- `e_fields_tup`: A tuple of preallocated target grids for registered statistical fields.
- `ldata::LSimData`: The source Lagrangian dataset.
"""
function interpolate_to_grid!(
    ::Val{:scatter},
    u_euler, w_euler, e_fields_tup,
    ldata::LSimData{D, DS, M, T}, 
    field_vals_tup, nan_vec,
    s_mins, s_maxs, s_dx, s_inv_dx, grid_shape
) where {D, DS, M, T}
    T_len = length(ldata.t)
    
    # --- Pre-calculate smoothing metrics ---
    N_p_initial = max(1, length(ldata.x[1]))
    pts_per_dim = max(1.0, N_p_initial^(1 / DS) - 1.0)
    particle_spacings = SVector{DS, T}([(s_maxs[d] - s_mins[d]) / pts_per_dim for d in 1:DS])
    effective_spacing = max.(s_dx, particle_spacings)
    radius = (norm(effective_spacing) * 1.5)^2
    radius_1d = sqrt(radius) 

    # =========================================================================
    # TIME BATCH LOOP (Dynamic Scatter Algorithm)
    # =========================================================================
    Threads.@threads for t_idx in 1:T_len
        x_step = ldata.x[t_idx]
        u_step = ldata.u[t_idx]
        N_p = length(x_step)
        
        # Pass 1: Scatter particles and fields to local grid cells
        @inbounds for p_idx in 1:N_p
            pos = x_step[p_idx]
            
            idx_float = (pos .- s_mins) .* s_inv_dx .+ 1.0
            rad_idx = radius_1d .* s_inv_dx
            
            min_idx = ntuple(d -> max(1, floor(Int, idx_float[d] - rad_idx[d])), Val(DS))
            max_idx = ntuple(d -> min(grid_shape[d], ceil(Int, idx_float[d] + rad_idx[d])), Val(DS))
            
            for cell_idx in CartesianIndices(ntuple(d -> min_idx[d]:max_idx[d], Val(DS)))
                s_idx = SVector{DS, T}(Tuple(cell_idx))
                cell_pos = s_mins + s_dx .* (s_idx .- 1.0)
                
                dist2 = sum(abs2.(cell_pos .- pos)) 
                
                if dist2 <= radius
                    w = 1.0 / max(dist2, 1e-12) 
                    
                    if D > DS
                        full_idx = CartesianIndex(Tuple(cell_idx)..., t_idx)
                        
                        w_euler[full_idx] += w
                        u_euler[full_idx] += u_step[p_idx] * w
                        
                        for i in 1:length(e_fields_tup)
                            e_fields_tup[i][full_idx] += field_vals_tup[i][t_idx][p_idx] * w
                        end
                    else
                        w_euler[cell_idx] += w
                        u_euler[cell_idx] += u_step[p_idx] * w
                        for i in 1:length(e_fields_tup)
                            e_fields_tup[i][cell_idx] += field_vals_tup[i][p_idx] * w
                        end
                    end
                end
            end
        end
        
        # Pass 2: Finalize averages for this time step
        @inbounds for cell_idx in CartesianIndices(grid_shape)
            if D > DS
                full_idx = CartesianIndex(Tuple(cell_idx)..., t_idx)
                w_sum = w_euler[full_idx]
                if w_sum > 0.0
                    u_euler[full_idx] /= w_sum
                    for i in 1:length(e_fields_tup)
                        e_fields_tup[i][full_idx] /= w_sum
                    end
                else
                    u_euler[full_idx] = nan_vec
                    for i in 1:length(e_fields_tup)
                        e_fields_tup[i][full_idx] = nan_vec
                    end
                end
            else
                w_sum = w_euler[cell_idx]
                if w_sum > 0.0
                    u_euler[cell_idx] /= w_sum
                    for i in 1:length(e_fields_tup)
                        e_fields_tup[i][cell_idx] /= w_sum
                    end
                else
                    u_euler[cell_idx] = nan_vec
                    for i in 1:length(e_fields_tup)
                        e_fields_tup[i][cell_idx] = nan_vec
                    end
                end
            end
        end
    end
end

"""
    resample_eulerian(data::ESimData, res::NTuple{D, Int})

Resamples the full Eulerian spacetime tensor, as well as all dimensionally-dependent custom statistics, to a new specified grid resolution. 
Crucially, it utilizes linear interpolation for spatial dimensions and constant (nearest-neighbor) interpolation for the time dimension, which prevents artifacting and cross-fading on temporal data.

# Arguments
- `data::ESimData`: The original Eulerian dataset to be resampled.
- `res::NTuple{D, Int}`: The exact target spacetime resolution (e.g., `(100, 100, 50)`).

# Returns
- `ESimData`: The newly interpolated Eulerian dataset, wrapped with updated `DomainInfo` and spacing.
"""
function resample_eulerian(data::ESimData{D, DS, M, T}, res::NTuple{D, Int}) where {D, DS, M, T}
    if size(data.u) == res
        return data
    end
    
    @info "Interpolating Eulerian data from $(size(data.u)) to$res (Linear Space, Constant Time)..."
    
    new_axes = ntuple(D) do d
        collect(range(data.domain.mins[d], data.domain.maxs[d], length=res[d]))
    end
    
    interp_types = ntuple(Val(D)) do d
        data.domain.dim_keys[d] == data.domain.time_dim ? Gridded(Constant()) : Gridded(Linear())
    end
    
    itp_obj = interpolate(data.axes, data.u, interp_types)
    itp = extrapolate(itp_obj, Flat())
    
    new_u = [itp(pt...) for pt in Iterators.product(new_axes...)]
    
    new_stats = StatDict{M, T}()
    for (k, v) in data.stats
        if !(v isa AbstractArray) || isempty(v)
            new_stats[k] = copy(v)
            continue
        end
        
        kept_dims = get_kept_dims(k, data.domain)
        if isempty(kept_dims)
            new_stats[k] = copy(v)
            continue
        end
        
        kept_indices = get_kept_indices(k, data.domain)
        stat_res = ntuple(i -> res[kept_indices[i]], length(kept_indices))
        
        if size(v) == stat_res
            new_stats[k] = copy(v)
            continue
        end
        
        stat_axes = ntuple(i -> data.axes[kept_indices[i]], length(kept_indices))
        stat_new_axes = ntuple(i -> new_axes[kept_indices[i]], length(kept_indices))
        
        stat_interp_types = ntuple(length(kept_indices)) do i
            data.domain.dim_keys[kept_indices[i]] == data.domain.time_dim ? Gridded(Constant()) : Gridded(Linear())
        end
        
        stat_itp_obj = interpolate(stat_axes, v, stat_interp_types)
        stat_itp = extrapolate(stat_itp_obj, Flat())
        
        new_stats[k] = [stat_itp(pt...) for pt in Iterators.product(stat_new_axes...)]
    end
    
    new_spacing = SVector{D, T}(
        ntuple(d -> res[d] > 1 ? (data.domain.maxs[d] - data.domain.mins[d]) / T(res[d] - 1) : one(T), Val(D))
    )
    
    new_domain = DomainInfo{D, T}(
        data.domain.dim_keys, data.domain.mins, data.domain.maxs, new_spacing, 
        data.domain.time_dim, data.domain.stat_registry
    )
    
    return ESimData{D, DS, M, T}(data.params, new_domain, new_axes, new_u, new_stats)
end

"""
    convert_to_eulerian(ldata::LSimData, res::NTuple{D, Int}; spatial_interp=:scatter)

Translates scattered Lagrangian particle data into a structured Eulerian tensor at a specified resolution. 
It maps all relevant statistical fields using the scatter algorithm, handles coordinate normalization, and conditionally routes the output through a temporal resampling pass if the target time frames differ from the native data.

# Arguments
- `ldata::LSimData`: The native Lagrangian dataset.
- `res::NTuple{D, Int}`: The requested spacetime grid resolution.

# Keyword Arguments
- `spatial_interp::Symbol`: The interpolation algorithm to use (defaults to `:scatter`).

# Returns
- `ESimData`: The synthesized Eulerian dataset.
"""
function convert_to_eulerian(
    ldata::LSimData{D, DS, M, T}, 
    res::NTuple{D, Int}; 
    spatial_interp::Symbol=:scatter
) where {D, DS, M, T}
    mins, maxs = ldata.domain.mins, ldata.domain.maxs
    time_dim = ldata.domain.time_dim
    t_dim_idx = get_time_dim(ldata.domain)
    T_len = length(ldata.t)
    
    spatial_res = isnothing(t_dim_idx) ? res : ntuple(d -> res[d < t_dim_idx ? d : d+1], Val(DS))
    
    grid_axes = ntuple(d -> collect(range(mins[d], maxs[d], length=spatial_res[d])), Val(DS))
    grid_shape = spatial_res
    
    s_mins = SVector{DS, T}(mins[1:DS])
    s_maxs = SVector{DS, T}(maxs[1:DS])
    s_dx = (s_maxs - s_mins) ./ max.(1, spatial_res .- 1)
    s_inv_dx = 1.0 ./ s_dx
    
    e_shape = ntuple(d -> d <= DS ? spatial_res[d] : T_len, Val(D))
    e_axes = ntuple(d -> d <= DS ? grid_axes[d] : ldata.t, Val(D))
    e_dim_keys = D > DS ? (ldata.domain.dim_keys[1:DS]..., time_dim) : ldata.domain.dim_keys
    e_spacing = SVector{D, T}(ntuple(d -> d <= DS ? s_dx[d] : ldata.domain.spacing[d], Val(D)))
    e_domain = DomainInfo{D, T}(e_dim_keys, mins, maxs, e_spacing, time_dim, ldata.domain.stat_registry)

    zero_vec = zero(SVector{M, T})
    nan_vec = zero_vec .* NaN
    
    u_euler = fill(zero_vec, e_shape...)
    w_euler = zeros(T, e_shape...)
    
    e_stats = StatDict{M, T}()
    field_keys = Symbol[]
    
    e_fields_vec = Array{SVector{M, T}, D}[]
    typeof_field_vals = D > DS ? Vector{Vector{SVector{M, T}}} : Vector{SVector{M, T}}
    field_vals_vec = typeof_field_vals[]
    
    for (k, v) in ldata.stats
        if k === :Solution; continue end
        kept_dims = get_kept_dims(k, ldata.domain)
        
        if kept_dims == [time_dim] || isempty(kept_dims)
            e_stats[k] = copy(v)
        else
            push!(field_keys, k)
            push!(e_fields_vec, fill(zero_vec, e_shape...))
            push!(field_vals_vec, v)
        end
    end

    e_fields_tup = Tuple(e_fields_vec)
    field_vals_tup = Tuple(field_vals_vec)

    interpolate_to_grid!(
        Val(spatial_interp), u_euler, w_euler, e_fields_tup, ldata, 
        field_vals_tup, nan_vec, 
        s_mins, s_maxs, s_dx, s_inv_dx, grid_shape
    )

    for i in 1:length(field_keys)
        e_stats[field_keys[i]] = e_fields_vec[i]
    end

    e_stats[:Solution] = u_euler

    edata = ESimData{D, DS, M, T}(ldata.params, e_domain, e_axes, u_euler, e_stats)
    
    needs_time_resampling = D > DS && T_len != res[t_dim_idx]
    return needs_time_resampling ? resample_eulerian(edata, res) : edata
end

"""
    convert_to_lagrangian(data::ESimData)

Translates a dense Eulerian grid into static, structured Lagrangian particles across time. 
It maps all custom statistical fields to the new particle arrays, flattening multidimensional tensor slices to align with the spatial coordinates at every time step.

# Arguments
- `data::ESimData`: The native Eulerian dataset.

# Returns
- `LSimData`: The synthesized Lagrangian dataset representing the grid as stationary particles, with all spatial fields perfectly preserved.
"""
function convert_to_lagrangian(data::ESimData{D, DS, M, T}) where {D, DS, M, T}
    t_dim = get_time_dim(data.domain)
    is_static = isnothing(t_dim)
    
    T_len = is_static ? 1 : length(data.axes[t_dim])
    t_vec = is_static ? T[0.0] : data.axes[t_dim]
    
    spatial_dims = Tuple(filter(d -> d != t_dim, 1:D))
    grid_shape = ntuple(i -> length(data.axes[spatial_dims[i]]), DS)
    
    pts = vec([SVector{DS, T}(ntuple(i -> data.axes[spatial_dims[i]][idx[i]], Val(DS))) 
               for idx in CartesianIndices(grid_shape)])
    
    new_x = [copy(pts) for _ in 1:T_len]
    new_u = Vector{Vector{SVector{M, T}}}(undef, T_len)
    
    for t in 1:T_len
        slice = is_static ? data.u : selectdim(data.u, t_dim, t)
        # Using collect ensures SubArrays are manifested into flat Vectors
        new_u[t] = collect(vec(slice)) 
    end
    
    new_stats = StatDict{M, T}()
    for (k, v) in data.stats
        if k === :Solution
            continue 
        end
        
        kept_dims = get_kept_dims(k, data.domain)
        
        if kept_dims == [data.domain.time_dim] || isempty(kept_dims)
            new_stats[k] = copy(v)
        else
            has_time = !is_static && (data.domain.time_dim in kept_dims)
            
            ET = eltype(v)
            new_field = Vector{Vector{ET}}(undef, T_len)
            
            for t in 1:T_len
                if has_time
                    stat_t_dim = findfirst(x -> x == data.domain.time_dim, kept_dims)
                    slice = selectdim(v, stat_t_dim, t)
                    # Using collect prevents ReshapedArray assignment errors
                    new_field[t] = collect(vec(slice)) 
                else
                    new_field[t] = collect(vec(v))
                end
            end
            new_stats[k] = new_field
        end
    end
    
    new_stats[:Solution] = new_u
    
    return LSimData{D, DS, M, T}(
        data.params, data.domain, t_vec, new_x, new_u, new_stats
    )
end