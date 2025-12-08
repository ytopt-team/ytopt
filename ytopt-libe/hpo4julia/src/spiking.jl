include("gpu.jl")

function bias_current(bias::LuxParams, t::Real, t_offset::Real, spk_args::SpikingArgs)
    bias = bias.bias_real .+ 1im .* bias.bias_imag
    return bias_current(bias, t, t_offset, spk_args)
end

function bias_current(bias::AbstractArray{<:Complex}, t::Real, t_offset::Real, spk_args::SpikingArgs)
    phase = complex_to_angle(bias)
    mag = abs.(bias)
    return bias_current(phase, mag, t, t_offset, spk_args)
end

function bias_current(phase::AbstractArray{<:Real}, mag::AbstractArray{<:Real}, t::Real, t_offset::Real, spk_args::SpikingArgs)
    #what times to the bias values correlate to?
    times = phase_to_time(phase, spk_args=spk_args, offset=t_offset)
    #determine the time within the cycle
    t = mod(t, spk_args.t_period)

    #add the active currents, scaled by the gaussian kernel & bias magnitude
    current_kernel = x -> gaussian_kernel(x, t, spk_args.t_window)
    bias = mag .* current_kernel(times)

    return bias
end

function check_offsets(x::SpikeTrain, y::SpikeTrain)
    if x.offset != y.offset
        return false
    else
        return true
    end
end

function check_offsets(x::SpikeTrain...)
    offset = x[1].offset
    for st in x
        if st.offset != offset
            return false
        end
    end
    return true
end

function count_nans(phases::Array{<:Real,3})
    return mapslices(x->sum(isnan.(x)), phases, dims=(2,3)) |> vec
end

function delay_train(train::SpikingTypes, t::Real, offset::Real)
    times = train.times .+ t

    if typeof(train) == SpikeTrain
        new_train = SpikeTrain(train.indices, times, train.shape, train.offset + offset)
    else
        new_train = SpikeTrainGPU(train.indices, times, train.shape, train.offset + offset)
    end

    return new_train
end

function end_phase((u, t)::Tuple; spk_args::SpikingArgs, offset::Real=0.0f0, kwargs...)
    phase = potential_to_phase(u[end], t[end], offset=offset, spk_args=spk_args; kwargs...)

    return phase
end

function find_spikes_rf(sol::ODESolution, spk_args::SpikingArgs; dim::Int=-1)
    @assert typeof(sol.u) <: Vector{<:Array{<:Complex}} "This method is for R&F neurons with complex potential"    
    t = sol.t
    u = solution_to_potential(sol, t)

    return find_spikes_rf(u, t, spk_args, dim=dim)
end

function find_spikes_rf(u::AbstractArray, t::AbstractVector, spk_args::SpikingArgs; dim::Int=-1)
    #choose the last dimension as default
    if dim == -1
        dim = ndims(u)
    end

    #if potential is from an R&F neuron, it is complex and voltage is the imaginary part
    voltage = imag.(u)
    current = real.(u)

    #find the local voltage maxima through the first derivative (current)
    op = x -> x .< 0.0f0
    #find maxima along the temporal dimension
    maxima = findall(op(diff(sign.(current), dims=dim)))
    peak_voltages = voltage[maxima]
    #check voltages at these peaks are above the threshold
    above_threshold = peak_voltages .> spk_args.threshold
    spikes = maxima[above_threshold]

    #retrieve the indices of the spiking neurons
    ax = 1:ndims(u) |> collect
    spatial_ax = setdiff(ax, dim)
    spatial_idx = [getindex.(spikes, i) for i in spatial_ax]
    channels = CartesianIndex.(spatial_idx...) 
    #retrieve the times they spiked at
    times = t[getindex.(spikes, dim)]
    
    return channels, times
end

function gaussian_kernel(x::AbstractArray, t::Real, t_sigma::Real)
    i = exp.(-1.0f0 .* ((t .- x) / (2.0f0 .* t_sigma)).^2.0f0)
    return i
end

function gaussian_kernel_vec(x::AbstractVector, ts::Vector, t_sigma::Real)
    i = exp.(-1.0f0 .* ((ts' .- x) / (2.0f0 .* t_sigma)).^2.0f0)
    return i
end

function arc_gaussian_kernel(x::AbstractVecOrMat, t::Real, t_sigma::Real)
    i = exp.(-1.0f0 .* (sin.(0.5f0 * pi_f32 .* (t .- x)) / (2.0f0 .* t_sigma)).^2.0f0)
    return i
end

function generate_cycles(tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs, offset::Real)
    #determine what the cycle offset should be
    offset = mod(offset, spk_args.t_period)
    #determine when each cycles begins and ends
    r = tspan[1]:spk_args.t_period:tspan[2]
    r = collect(r) .+ offset
    return r[2:end]
end

function is_active(times::AbstractArray, t::Real, t_window::Real; sigma::Real=9.0f0)
    active = (times .> (t - sigma * t_window)) .* (times .< (t + sigma * t_window))
    return active
end

"""
Delay spike trains as necessary to make the represented phases between them match
"""
function match_offsets(x::SpikingTypes, y::SpikingTypes)
    xo = x.offset
    yo = y.offset

    if xo == yo
        return x, y
    elseif xo > yo
        dy = xo - yo
        yp = delay_train(y, dy, dy)
        return x, yp
    else
        dx = yo - xo
        xp = delay_train(x, dx, dx)
        return xp, y
    end
end

"""
Delay the spike trains in a vector as necessary to make their offsets match
"""
function match_offsets(x::Vector{<:SpikingTypes})
    offsets = getfield.(x, :offset)
    final = maximum(offsets)
    dt = final .- offsets
    new_trains = [delay_train(st, dt[i], dt[i]) for (i, st) in enumerate(x)]
    return new_trains
end

function match_tspans(spans::Tuple{<:Real, <:Real}...)
    start = minimum([s[1] for s in spans])
    stop = maximum([s[2] for s in spans])
    return (start, stop)
end

function mean_phase((u, t)::Tuple, warmup::Real; spk_args::SpikingArgs, offset::Real=0.0f0, kwargs...)
    inds = t .> spk_args.t_period * warmup
    phase = potential_to_phase((u[inds], t[inds]), offset=offset, spk_args=spk_args; kwargs...)
    phase = angular_mean(phase, dims=(3))[:,:,1]

    return phase
end

function normalize_potential(u::Complex)
    a = abs(u)
    if a == 0.0f0
        return u
    else
        return u / a
    end
end

function normalize_potential(a::AbstractArray)
    return normalize_potential.(a)
end

function spike_current(train::SpikeTrain, t::Real, spk_args::SpikingArgs; sigma::Real = 9.0f0)
    @assert typeof(spk_args.spike_kernel) <: Function || spk_args.spike_kernel == :gaussian "Unrecognized kernel type, defaulting to gaussian"
    current = zeros(Float32, train.shape)
    scale = spk_args.spk_scale

    ignore_derivatives() do
        #find which channels are active 
        times = train.times
        active = is_active(times, t, spk_args.t_window, sigma=sigma)
        active_inds = train.indices[active]
        active_tms = train.times[active]

        #add currents into the active synapses
        if typeof(spk_args.spike_kernel) <: Function
            current_kernel = x -> spk_args.spike_kernel(x, t)
        elseif spk_args.spike_kernel == :gaussian
            current_kernel = x -> gaussian_kernel(x, t, spk_args.t_window)
        end
        impulses = current_kernel(active_tms)
        
        current[active_inds] .+= (scale .*impulses)
    end

    return current
end

function spike_current(train::SpikeTrainGPU, t::Real, spk_args::SpikingArgs)
    scale = spk_args.spk_scale

    #add currents into the synapses
    current_kernel = x -> gaussian_kernel(x, t, spk_args.t_window)
    impulses = current_kernel(train.times)
    current = parallel_scatter_add(train.linear_indices, impulses, train.linear_shape)
    current = reshape(current, train.shape)
    
    return current
end

function spiking_offset(spk_args::SpikingArgs)
    return spk_args.t_period / 4.0f0
end

function stack_trains(trains::Array{<:SpikeTrain,1})
    check_offsets(trains...)
    n_t = length(trains)
    shape = trains[1].shape
    offset = trains[1].offset
    for t in trains
        @assert shape == t.shape "Spike trains must have identical shape to be stacked"
        @assert offset == t.offset "Spike trains must have identical offsets"
    end

    new_shape = (n_t, shape...)
    all_indices = []

    for (i, train) in enumerate(trains)
        old_indices = train.indices
        #add the new dimension for each index
        new_indices = [CartesianIndex((i, Tuple(idx)...)) for idx in old_indices]
        append!(all_indices, new_indices)
    end

    all_indices = vcat(all_indices...)
    all_times = reduce(vcat, collect(t.times for t in trains))

    new_train = SpikeTrain(all_indices, all_times, new_shape, offset)
    return new_train
end

function oscillator_bank(u0::AbstractArray, dzdt::Function; tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs)
    #solve the memory compartment
    prob = ODEProblem(dzdt, u0, tspan)
    sol = solve(prob, spk_args.solver; spk_args.solver_args...)
    
    return sol
end

function oscillator_bank(u0::AbstractArray, dzdt::Function, params::LuxParams; tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs)
    #solve the memory compartment with external parameters passed
    prob = ODEProblem(dzdt, u0, tspan, params)
    sol = solve(prob, spk_args.solver, p = params; spk_args.solver_args...)
    
    return sol
end

function oscillator_bank(x::CurrentCall, layer::AbstractLuxLayer, params::LuxParams, state::NamedTuple)
    return oscillator_bank(x.current, layer, params, state, tspan=x.t_span, spk_args=x.spk_args,)
end

function oscillator_bank(x::LocalCurrent, layer::AbstractLuxLayer, params::LuxParams, state::NamedTuple; tspan::Tuple{<:Real, <:Real} = (0.0f0, 10.0f0), spk_args::SpikingArgs, use_bias::Bool=true)
    #set up functions to define the neuron's differential equations
    update_fn = spk_args.update_fn
    #set up compartments for each sample
    output_sample = layer.layer(x.current_fn(0.0f0), params.layer, state.layer)[1]
    u0 = similar(output_sample, ComplexF32)
    ignore_derivatives() do
        u0 .= zero(ComplexF32) # Or ComplexF32(0.0f0)
    end

    #resonate in time with the input spikes, applying the kernel to the spike current
    function dzdt(u, p, t)
        transformed_current = layer.layer(x.current_fn(t), p.layer, state.layer)[1]
        biasing_current = bias_current(p.bias, t, x.offset, spk_args)
        return update_fn(u) .+ transformed_current .+ biasing_current
    end

    function dzdt_nobias(u, p, t)
        transformed_current = layer.layer(x.current_fn(t), p.layer, state.layer)[1]
        return update_fn(u) .+ transformed_current
    end

     #solve the memory compartment using the base oscillator_bank method
    if use_bias
        sol = oscillator_bank(u0, dzdt, tspan=tspan, spk_args=spk_args, params)
    else
        sol = oscillator_bank(u0, dzdt_nobias, tspan=tspan, spk_args=spk_args, params)
    end

    return sol
end

function oscillator_bank(x::CurrentCall)
    return oscillator_bank(x.current, tspan=x.t_span, spk_args=x.spk_args)
end

function oscillator_bank(x::LocalCurrent; tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs)
    #set up functions to define the neuron's differential equations
    update_fn = spk_args.update_fn

    #call the current function to find if we're on CPU or GPU
    sample = x.current_fn(tspan[1])
    #make the initial potential the bias value
    if typeof(sample) <: CuArray
        u0 = CUDA.zeros(ComplexF32, x.shape)
    else
        u0 = zeros(ComplexF32, x.shape)
    end
    
    #shift the solver span by the function's time offset
    tspan = tspan .+ x.offset

    #solve the ODE over the given time span
    dzdt(u, p, t) = update_fn(u) + x.current_fn(t)
    sol = oscillator_bank(u0, dzdt, tspan=tspan, spk_args=spk_args)

    return sol
end

function oscillator_bank(x::SpikingTypes; tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs)
    current_fn = t -> spike_current(x, t, spk_args)
    local_current = LocalCurrent(current_fn, x.shape, x.offset)
    return oscillator_bank(local_current, tspan=tspan, spk_args=spk_args)
end

# #special version used for ODE layers
# function oscillator_bank(x::LocalCurrent, params; tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs, return_solution::Bool=false)
#     #set up functions to define the neuron's differential equations
#     output_shape = (size(params.weight, 1), x.shape[2])
#     #make the initial potential the bias value
#     u0 = zeros(ComplexF32, output_shape)
#     #shift the solver span by the function's time offset
#     tspan = tspan .+ x.offset
    
#     #override spk args with params leakage and frequency if provided
#     function calc_k(p)
#         if haskey(p, :leakage) && haskey(p, :t_period)
#             angular_frequency = 2.0f0 * pi_f32 / p.t_period[1]
#             k = (p.leakage[1] + 1.0f0im * angular_frequency)
#         else
#             k = neuron_constant(spk_args)
#         end
#         return k
#     end
    
#     function dzdt(u, p, t)
#         k = calc_k(p)
#         du = k .* u + p.weight * x.current_fn(t) .+ bias_current(p.bias_real .+ 1im .* p.bias_imag, t, x.offset, spk_args)
#         return du
#     end

#     function dzdt_nobias(u, p, t)
#         k = calc_k(p)
#         du = k .* u + p.weight * x.current_fn(t)
#         return du
#     end
    
#     #enable bias if used
#     if haskey(params, :bias_real) && haskey(params, :bias_imag)
#         prob = ODEProblem(dzdt, u0, tspan, params)
#     else
#         prob = ODEProblem(dzdt_nobias, u0, tspan, params)
#     end
    
#     sol = solve(prob, spk_args.solver; spk_args.solver_args...)
#     return sol
# end

function vcat_trains(trains::Array{<:SpikingTypes,1})
    check_offsets(trains...)
    n_t = length(trains)
    shape = trains[1].shape
    offset = trains[1].offset
    for t in trains
        @assert shape == t.shape "Spike trains must have identical shape to be stacked"
        @assert offset == t.offset "Spike trains must have identical offsets"
    end

    new_shape = (n_t, shape[2:end]...)
    all_indices = []

    for (i, train) in enumerate(trains)
        old_indices = train.indices
        #add the new dimension for each index
        new_indices = [CartesianIndex((i, Tuple(idx)[2:end]...)) for idx in old_indices]
        append!(all_indices, new_indices)
    end

    all_indices = vcat(all_indices...)
    all_times = reduce(vcat, collect(t.times for t in trains))

    new_train = SpikeTrain(all_indices, all_times, new_shape, offset)
    return new_train
end

function zero_nans(phases::AbstractArray)
    nans = isnan.(phases)
    phases[nans] .= 0.0f0
    return phases
end