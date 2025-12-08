include("spiking.jl")

function v_bind(x::AbstractArray; dims)
    bz = sum(x, dims = dims)
    y = remap_phase(bz)
    return y
end

function v_bind(x::AbstractArray, y::AbstractArray)
    y = remap_phase(x .+ y)
    return y
end

function v_bind(x::Tuple{Vararg{AbstractArray}}; dims=1)
    x = cat((x...), dims=dims)
    return v_bind(x, dims=dims)
end

function v_bind(x::SpikingCall, y::SpikingCall; return_solution::Bool = false, unbind::Bool=false, automatch::Bool=true)
    output = v_bind(x.train, y.train; 
                tspan=x.t_span, 
                spk_args=x.spk_args,
                unbind=unbind,
                automatch=automatch,
                return_solution=return_solution)
    
    if return_solution
        return output
    end

    next_call = SpikingCall(output, x.spk_args, x.t_span)
    return next_call
end

function v_bind(x::SpikingTypes, y::SpikingTypes; tspan::Tuple{<:Real, <:Real} = (0.0f0, 10.0f0), spk_args::SpikingArgs, return_solution::Bool = false, unbind::Bool=false, automatch::Bool=true)
    if !automatch
        if check_offsets(x::SpikingTypes, y::SpikingTypes) @warn "Offsets between spike trains do not match - may not produce desired phases" end
    else
        x, y = match_offsets(x, y)
    end

    #set up functions to define the neuron's differential equations
    k = neuron_constant(spk_args)

    #get the number of batches & output neurons
    output_shape = x.shape

    #find the complex state induced by the spikes
    sol_x = oscillator_bank(x, tspan=tspan, spk_args=spk_args)
    sol_y = oscillator_bank(y, tspan=tspan, spk_args=spk_args)
    
    #create a reference oscillator to generate complex values for each moment in time
    u_ref = t -> phase_to_potential(0.0f0, t, offset = x.offset, spk_args = spk_args)

    #find the first chord
    chord_x = t -> sol_x(t)
    #find the second chord
    if unbind
        chord_y = t -> sol_x(t) .* conj.((sol_y(t) .- u_ref(t))) .* u_ref(t)
    else
        chord_y = t -> sol_x(t) .* (sol_y(t) .- u_ref(t)) .* conj(u_ref(t))
    end

    sol_output = t -> chord_x(t) .+ chord_y(t)
    
    if return_solution
        return sol_output
    end

    train = solution_to_train(sol_output, tspan, spk_args=spk_args, offset=x.offset)
    return train
end

function v_bundle(x::AbstractArray; dims::Int)
    xz = angle_to_complex(x)
    bz = sum(xz, dims = dims)
    y = complex_to_angle(bz)
    return y
end

function v_bundle(x::Tuple{Vararg{AbstractArray}}; dims=1)
    x = cat((x...), dims=dims)
    return v_bundle(x, dims=dims)
end

function v_bundle(x::SpikingCall; dims::Int)
    train = v_bundle(x.train, dims=dims, tspan=x.t_span, spk_args=x.spk_args)
    next_call = SpikingCall(train, x.spk_args, x.t_span)
    return next_call
end

function v_bundle(x::SpikingTypes; dims::Int, tspan::Tuple{<:Real, <:Real} = (0.0f0, 10.0f0), spk_args::SpikingArgs, return_solution::Bool=false)
    #let compartments resonate in sync with inputs
    sol = oscillator_bank(x, tspan=tspan, spk_args=spk_args)
    tbase = sol.t
    #combine the potentials (interfere) along the bundling axis
    f_sol = x -> sum(normalize_potential.(sol(x)), dims=dims)

    if return_solution
        return f_sol
    end
    
    out_train = solution_to_train(f_sol, tspan, spk_args=spk_args, offset=x.offset)
    return out_train
end

function v_bundle_project(x::AbstractArray, w::AbstractMatrix, b::AbstractVecOrMat)
    xz = batched_mul(w, angle_to_complex(x)) .+ b
    #y = complex_to_angle(xz)
    y = soft_angle(xz, 0.01f0, 0.1f0)
    return y
end

function v_bundle_project(x::SpikingCall, w::AbstractMatrix, b::AbstractVecOrMat; return_solution::Bool=false)
    sol = oscillator_bank(x.train, w, b, tspan=x.t_span, spk_args=x.spk_args)
    if return_solution
        return sol
    end

    train = solution_to_train(sol, x.t_span, spk_args=x.spk_args, offset=x.train.offset)
    next_call = SpikingCall(train, x.spk_args, x.t_span)
    return next_call
end

function v_bundle_project(x::CurrentCall, w::AbstractMatrix, b::AbstractVecOrMat; return_solution::Bool=false)
    sol = oscillator_bank(x.current, w, b, tspan=x.t_span, spk_args=x.spk_args)
    if return_solution
        return sol
    end
    
    train = solution_to_train(sol, x.tspan, spk_args=x.spk_args, offset=x.offset)
    next_call = SpikingCall(train, x.spk_args, x.t_span)
    return next_call
end

function v_bundle_project(x::CurrentCall, params; return_solution::Bool=false)
    sol = oscillator_bank(x.current, params, tspan=x.t_span, spk_args=x.spk_args)
    if return_solution
        return sol
    end
    
    train = solution_to_train(sol, x.tspan, spk_args=x.spk_args, offset=x.offset)
    next_call = SpikingCall(train, x.spk_args, x.t_span)
    return next_call
end

function chance_level(nd::Int, samples::Int)
    symbol_0 = random_symbols((1, nd))
    symbols = random_symbols((samples, nd))
    sim = similarity_outer(symbol_0, symbols, dims=1) |> vec
    dev = std(sim)

    return dev
end

function random_symbols(size::Tuple{Vararg{Int}})
    y = 2.0f0 .* rand(Float32, size) .- 1.0f0
    return y
end

function random_symbols(rng::AbstractRNG, size::Tuple{Vararg{Int}})
    y = 2.0f0 .* rand(rng, Float32, size) .- 1.0f0
    return y
end

function remap_phase(x::Real)
    ignore_derivatives() do
        x = x + 1.0f0
        x = mod(x, 2.0f0)
        x = x - 1.0f0
    end
    return x
end

function remap_phase(x::AbstractArray)
    ignore_derivatives() do
        x = x .+ 1.0f0
        x = mod.(x, 2.0f0)
        x = x .- 1.0f0
    end
    return x
end

function similarity(x::AbstractArray, y::AbstractArray; dim::Int = 1)
    if dim == -1
        dim = ndims(x)
    end

    dx = cos.(pi_f32 .* (x .- y))
    s = mean(dx, dims = dim)
    s = dropdims(s, dims = dim)
    return s
end

function similarity(x::SpikingTypes, y::SpikingTypes; spk_args::SpikingArgs, tspan::Tuple{<:Real, <:Real}, automatch::Bool=true)
    if !automatch
        if check_offsets(x::SpikingTypes, y::SpikingTypes) @warn "Offsets between spike trains do not match - may not produce desired phases" end
    else
        x, y = match_offsets(x, y)
    end

    sol_x = oscillator_bank(x, tspan = tspan, spk_args = spk_args)
    sol_y = oscillator_bank(y, tspan = tspan, spk_args = spk_args)

    u_x = normalize_potential.(Array(sol_x))
    u_y = normalize_potential.(Array(sol_y))

    interference = abs.(u_x .+ u_y)
    avg_sim = interference_similarity(interference, dim=1)
    return avg_sim
end

function interference_similarity(interference::AbstractArray; dim::Int=-1)
    if dim == -1
        dim = ndims(interference)
    end

    magnitude = clamp.(interference, 0.0f0, 2.0f0)
    half_angle = acos.(0.5f0 .* magnitude)
    sim = cos.(2.0f0 .* half_angle)
    avg_sim = mean(sim, dims=dim)
    avg_sim = dropdims(avg_sim, dims=dim)
    
    return avg_sim
end

function similarity_outer(x::SpikingCall, y::SpikingCall; automatch::Bool=true)
    @assert x.spk_args == y.spk_args "Spiking arguments must be identical to calculate similarity"
    new_span = match_tspans(x.t_span, y.t_span)
    return similarity_outer(x.train, y.train, tspan=new_span, spk_args=x.spk_args, automatch=automatch)
end

function similarity_outer(x::SpikingTypes, y::SpikingTypes; tspan::Tuple{<:Real, <:Real} = (0.0f0, 10.0f0), spk_args::SpikingArgs, automatch::Bool=true)
    # Allow arbitrary dimensionality for spike-train batches. We will slice along
    # the last two dimensions (batch x features) by default in downstream
    # similarity_outer for arrays, so no strict shape assertion is required here.
    if !automatch
        if check_offsets(x::SpikingTypes, y::SpikingTypes) @warn "Offsets between spike trains do not match - may not produce desired phases" end
    else
        x, y = match_offsets(x, y)
    end

    sol_x = oscillator_bank(x, tspan = tspan, spk_args = spk_args)
    sol_y = oscillator_bank(y, tspan = tspan, spk_args = spk_args)

    u_x = normalize_potential.(sol_x.u)
    u_y = normalize_potential.(sol_y.u)
    
    #add up along the slices
    sim = similarity_outer.(u_x, u_y)
    return sim
end

function similarity_outer(x::CurrentCall, y::CurrentCall)
    @assert x.spk_args == y.spk_args "Spiking arguments must be identical to calculate similarity"
    new_span = match_tspans(x.t_span, y.t_span)

    sol_x = oscillator_bank(x)
    sol_y = oscillator_bank(y)

    u_x = normalize_potential.(sol_x.u)
    u_y = normalize_potential.(sol_y.u)
    
    #add up along the slices
    return similarity_outer.(u_x, u_y)
end

function similarity_self(x::AbstractArray; dims)
    return similarity_outer(x, x, dims=dims)
end

"""
Slicing each array along 'dims', find the similarity between each corresponding slice and
reduce along 'reduce_dim'
"""
function similarity_outer(x::AbstractArray{<:Real,3}, y::AbstractArray{<:Real,3}; dims=2)
    s = [similarity(xs, ys) for xs in eachslice(x, dims=dims), ys in eachslice(y, dims=dims)]
    #stack and reshape to batch-last
    s = permutedims(stack(s), (2,3,1))
    return s
end

function similarity_outer(x::AbstractArray{<:Real,2}, y::AbstractArray{<:Real,2}; dims=2)
    s = [similarity(xs, ys) for xs in eachslice(x, dims=dims), ys in eachslice(y, dims=dims)]
    #stack and reshape to batch-last
    s = permutedims(stack(s), (2,1))
    return s
end

function similarity_outer(x::AbstractArray{<:Complex}, y::AbstractArray{<:Complex}; dims=2)
    s = [interference_similarity(abs.(xs .+ ys), dim=dims) for xs in eachslice(x, dims=dims), ys in eachslice(y, dims=dims)]
    #stack and reshape to batch-last
    s = permutedims(stack(s), (2,3,1))
    return s
end

#Note - additional definitions for similarity_outer included in gpu.jl

function v_unbind(x::AbstractArray, y::AbstractArray)
    y = remap_phase(x .- y)
    return y
end

function v_unbind(x::SpikingTypes, y::SpikingTypes; kwargs...)
    return v_bind(x, y, unbind=true; kwargs...)
end