include("types.jl")

function angle_to_complex(x::AbstractArray)
    k = pi_f32 * (0.0f0 + 1.0f0im)
    return exp.(k .* x)
end

function complex_to_angle(x::AbstractArray)
    return angle.(x) ./ pi_f32
end

function complex_to_angle(x_real::Real, x_imag::Real)
    return atan(x_imag, x_real) / pi_f32
end

function soft_angle(x::AbstractArray{<:Complex}, r_lo::Real = 0.1f0, r_hi::Real = 0.2f0)
    s = similar(real.(x))

    ignore_derivatives() do
        r = abs.(x)
        m = (r .- r_lo) ./ (r_hi - r_lo)
        s .= sigmoid_fast(3.0f0 .* m .- (r_hi - r_lo))
    end

    return s .* angle.(x) / pi_f32
end


function cmpx_to_realvec(u::Array{<:Complex})
    nd = ndims(u)
    reals = real.(u)
    imags = imag.(u)
    mat = stack((reals, imags), dims=1)
    return mat
end

function realvec_to_cmpx(u::Array{<:Real})
    @assert size(u)[1] == 2 "Must have first dimension contain real and imaginary values"
    slices = eachslice(u, dims=1)
    mat = slices[1] .+ 1.0f0im .* slices[2]
    return mat
end

###
### PHASE - SPIKE
###

"""
Converts a matrix of phases into a spike train via phase encoding

phase_to_train(phases::AbstractMatrix, spk_args::SpikingArgs, repeats::Int = 1, offset::Real = 0.0)
"""
function phase_to_time(phases::AbstractArray; offset::Real = 0.0f0, spk_args::SpikingArgs)
    return phase_to_time(phases, spk_args.t_period, Float32(offset))
end

function phase_to_time(phases::AbstractArray, period::Real, offset::Real = 0.0f0)
    phases = eltype(phases) == Float32 ? phases : Float32.(phases)
    period = Float32(period)
    offset = Float32(offset)
    #convert a potential to the time at which the voltage is maximum - 90* behind phase
    phases = (phases ./ 2.0f0) .+ 0.5f0
    times = phases .* period .+ offset
    #make all times positive
    times = mod.(times, period)
   
    return times
end

function time_to_phase(times::AbstractArray; spk_args::SpikingArgs, offset::Real)
    return time_to_phase(times, spk_args.t_period, offset)
end

function time_to_phase(times::AbstractArray, period::Real, offset::Real)
    times = mod.((times .- offset), period) ./ period
    phase = (times .- 0.5f0) .* 2.0f0
    return phase
end

function phase_to_train(phases::AbstractArray; spk_args::SpikingArgs, repeats::Int = 1, offset::Real = 0.0f0)
    shape = phases |> size
    indices = collect(CartesianIndices(shape)) |> vec
    times = phase_to_time(phases, spk_args=spk_args, offset=offset) |> vec

    if repeats > 1
        n_t = times |> length
        offsets = repeat(collect(0:repeats-1) .* spk_args.t_period, inner=n_t)
        times = repeat(times, repeats) .+ offsets
        indices = repeat(indices, repeats)
    end

    train = SpikeTrain(indices, times, shape, offset)
    return train
end

function train_to_phase(call::SpikingCall)
    return train_to_phase(call.train, spk_args=call.spk_args)
end

function train_to_phase(train::SpikeTrainGPU; spk_args::SpikingArgs)
    train = SpikeTrain(train)
    #preserve device on output
    phases = train_to_phase(train, spk_args=spk_args, offset=train.offset) |> gdev
    return phases
end

function train_to_phase(train::SpikeTrain; spk_args::SpikingArgs, offset::Real = 0.0f0)
    if length(train.times) == 0
        return missing
    end

    @assert reduce(*, train.times .>= 0.0f0) "Spike train times must be positive"

    #decode each spike's phase within a cycle
    relative_phase = time_to_phase(train.times, spk_args.t_period, train.offset)
    relative_time = train.times .- (train.offset + offset)
    #what is the cycle in which each spike occurs?
    cycle = floor.(Int, relative_time .÷ spk_args.t_period)
    #re-number cycles to be positive
    cycle = cycle .+ (1 - minimum(cycle))
    #what is the number of cycles in this train?
    n_cycles = maximum(cycle)
    phases = [fill(Float32(NaN), train.shape...) for i in 1:n_cycles]

    for i in eachindex(relative_phase)
        phases[cycle[i]][train.indices[i]] = relative_phase[i]
    end

    #stack the arrays to batch, neuron, cycle
    phases = mapreduce(x->reshape(x, train.shape..., 1), (a,b)->cat(a, b, dims=ndims(a)), phases)
    return phases
end

function phase_to_current(phases::AbstractArray; spk_args::SpikingArgs, offset::Real = 0.0f0, tspan::Tuple{<:Real, <:Real}, rng::Union{AbstractRNG, Nothing} = nothing, zeta::Real=Float32(0.0))
    shape = size(phases)
    
    function inner(t::Real)
        output = similar(phases)

        ignore_derivatives() do
            p = time_to_phase([t,], spk_args = spk_args, offset = offset)[1]
            current_kernel = x -> arc_gaussian_kernel(x, p, spk_args.t_window * period_to_angfreq(spk_args.t_period))
            impulses = current_kernel(phases)

            if zeta > 0.0f0
                noise = zeta .* randn(rng, Float32, size(impulses))
                impulses .+= noise
            end
            
            output .= impulses
        end

        return output
    end

    current = LocalCurrent(inner, shape, offset)
    call = CurrentCall(current, spk_args, tspan)

    return call
end

###
### PHASE - POTENTIAL
###

"""
Convert a static phase to the complex potential of an R&F neuron
"""
function phase_to_potential(phase::Real, ts::AbstractVector; offset::Real=0.0f0, spk_args::SpikingArgs)
    return [phase_to_potential(phase, t, offset=offset, spk_args=spk_args) for t in ts]
end

function phase_to_potential(phase::AbstractArray, ts::AbstractVector; offset::Real=0.0f0, spk_args::SpikingArgs)
    return [phase_to_potential(p, t, offset=offset, spk_args=spk_args) for p in phase, t in ts]
end

function phase_to_potential(phase::Real, t::Real; offset::Real=0.0f0, spk_args::SpikingArgs)
    period = Float32(spk_args.t_period)
    k = ComplexF32(1.0f0im * imag(neuron_constant(spk_args)))
    potential = ComplexF32(exp.(k .* ((t .- offset) .- (phase - 1.0f0)/2.0f0 * period)))
    return potential
end

"""
Convert the potential of a neuron at an arbitrary point in time to its phase relative to a reference
"""
function potential_to_phase(potential::AbstractArray, t::Real; offset::Real=0.f0, spk_args::SpikingArgs, threshold::Bool=false)
    current_zero = similar(potential, ComplexF32, (1))

    ignore_derivatives() do
        #find the angle of a neuron representing 0 phase at the current moment in time
        current_zero = phase_to_potential(0.0f0, t, offset=offset, spk_args=spk_args)
    end
    #get the arc subtended in the complex plane between that reference and our neuron potentials
    arc = angle(current_zero) .- angle.(potential) 

    #normalize by pi and shift to -1, 1
    phase = mod.((arc ./ pi_f32 .+ 1.0f0), 2.0f0) .- 1.0f0

    #replace silent neurons with NaN values
    ignore_derivatives() do
        if threshold
            silent = findall(abs.(potential) .<= spk_args.threshold)
            for i in silent
                phase[i] = Float32(NaN)
            end
        end
    end

    return phase
end

"""
    potential_to_phase(ut::Tuple{<:AbstractVector{<:AbstractArray}, <:AbstractVector}; spk_args::SpikingArgs, kwargs...)

Decodes the phase from a tuple of potentials and times, as produced by an `ODESolution`.
This is a convenience function for handling the output of ODE solvers like `(sol.u, sol.t)`.
"""
function potential_to_phase(ut::Tuple{<:AbstractVector{<:AbstractArray}, <:AbstractVector}; spk_args::SpikingArgs, kwargs...)
    u_vec = ut[1]
    ts = ut[2]

    # Stack the vector of arrays into a single multi-dimensional array, adding a time dimension.
    potential = stack(u_vec, dims=ndims(u_vec[1]) + 1)

    return potential_to_phase(potential, ts; spk_args=spk_args, kwargs...)
end

function potential_to_phase(potential::AbstractArray, ts::AbstractVector; spk_args::SpikingArgs, offset::Real=0.0f0, threshold::Bool=false)
    @assert size(potential)[end] == length(ts) "Time dimensions must match"
    current_zeros = similar(potential, ComplexF32, (length(ts)))
    dims = collect(1:ndims(potential))

    ignore_derivatives() do
        #find the angle of a neuron representing 0 phase at the current moment in time
        current_zeros = phase_to_potential.(0.0f0, ts, offset=offset, spk_args=spk_args)
    end
    #get the arc subtended in the complex plane between that reference and our neuron potentials
    potential = permutedims(potential, reverse(dims))
    arc = angle.(current_zeros) .- angle.(potential) 
    
    #normalize by pi and shift to -1, 1
    phase = mod.((arc ./ pi_f32 .+ 1.0f0), 2.0f0) .- 1.0f0

    #replace silent neurons with random values
    ignore_derivatives() do
        if threshold
            silent = findall(abs.(potential) .<= spk_args.threshold)
            for i in silent
                phase[i] = Float32(NaN)
            end
        end
    end

    phase = permutedims(phase, reverse(dims))
    return phase
end

function solution_to_potential(func_sol::Union{ODESolution, Function}, t::Array)
    u = func_sol.(t)
    d = ndims(u[1])
    #stack the vector of solutions along a new final axis
    u = stack(u, dims = d + 1)
    return u
end

function solution_to_potential(ode_sol::ODESolution)
    return Array(ode_sol)
end

function solution_to_phase(sol::ODESolution; final_t::Bool=false, offset::Real=0.0f0, spk_args::SpikingArgs, kwargs...)
    #convert the ODE solution's saved points to an array
    u = solution_to_potential(sol)
    if final_t
        u = u[:,:,end]
        p = potential_to_phase(u, sol.t[end], offset=offset, spk_args=spk_args; kwargs...)
    else
        #calculate the phase represented by that potential
        p = potential_to_phase(u, sol.t, offset=offset, spk_args=spk_args; kwargs...)
    end

    return p
end

function solution_to_phase(sol::Union{ODESolution, Function}, t::Array; offset::Real=0.0f0, spk_args::SpikingArgs, kwargs...)
    #call the solution at the provided times
    u = solution_to_potential(sol, t)
    #calculate the phase represented by that potential
    p = potential_to_phase(u, t, offset=offset, spk_args=spk_args; kwargs...)
    return p
end

###
### POTENTIAL - TIME
###

function period_to_angfreq(t_period::Real)
    angular_frequency = 2.0f0 * pi_f32 / t_period
    return angular_frequency
end

function angfreq_to_period(angfreq::Real)
    #auto-inverting transform
    return period_to_angfreq(angfreq)
end

function neuron_constant(leakage::Real, t_period::Real)
    angular_frequency = period_to_angfreq(t_period)
    k = ComplexF32(leakage + 1.0f0im * angular_frequency)
    return k
end

function neuron_constant(spk_args::SpikingArgs)
    k = neuron_constant(spk_args.leakage, spk_args.t_period)
    return k
end

function potential_to_time(u::AbstractArray, t::Real; spk_args::SpikingArgs)
    spiking_angle = pi_f32 / 2.0f0

    #find out given this potential, how much time until the neuron spikes (ideally)
    angles = mod.(-1.0f0 .* angle.(u), 2.0f0*pi_f32) #flip angles and move onto the positive domain
    arc_to_spike = spiking_angle .+ angles
    time_to_spike = arc_to_spike ./ period_to_angfreq(spk_args.t_period)
    spikes = t .+ time_to_spike
    
    #make all times positive
    spikes[findall(x -> x < 0.0f0, spikes)] .+= spk_args.t_period
    return spikes
end

function potential_to_time(u::AbstractArray, ts::AbstractVector; spk_args::SpikingArgs, dim::Int=-1)
    if dim == -1
        dim = ndims(u)
    end
    @assert size(u, dim) == length(ts) "Time dimension of array must match list of times"

    u_slices = eachslice(u, dims=dim)
    spikes = [potential_to_time(x[1], x[2], spk_args=spk_args) for x in zip(u_slices, ts)]
    spikes = stack(spikes, dims=dim)
    return spikes
end

function time_to_potential(spikes::AbstractArray, t::Real; spk_args::SpikingArgs)
    spiking_angle = pi_f32 / 2.0f0

    #find out given this time, what is the (normalized) potential at a given moment?
    time_from_spike = spikes .- t
    arc_from_spike = time_from_spike .* period_to_angfreq(spk_args.t_period)
    angles = -1.0f0 .* (arc_from_spike .- spiking_angle)
    potentials = angle_to_complex(angles ./ pi_f32)

    return potentials
end

function time_to_potential(spikes::AbstractArray, ts::AbstractVector; spk_args::SpikingArgs, dim::Int=-1)
    if dim == -1
        dim = ndims(spikes)
    end
    @assert size(spikes, dim) == length(ts) "Time dimension of array must match list of times"

    t_slices = eachslice(spikes, dims=dim)
    potential = [time_to_potential(x[1], x[2], spk_args=spk_args) for x in zip(t_slices, ts)]
    potential = stack(potential, dims=dim)
    return potential
end

function solution_to_train(sol::Union{ODESolution,Function}, tspan::Tuple{<:Real, <:Real}; spk_args::SpikingArgs, offset::Real)
    #determine the ending time of each cycle
    cycles = generate_cycles(tspan, spk_args, offset)

    #sample the potential at the end of each cycle
    u = solution_to_potential(sol, cycles)
    train = solution_to_train(u, cycles, spk_args=spk_args, offset=offset)
    return train
end

"""
This implementation takes a full solution (represented by a vector of arrays) and finds the spikes from it.
"""
function solution_to_train(u::AbstractVector{<:AbstractArray}, t::AbstractVector{<:Real}, tspan::Tuple{<:Real, <:Real}; spk_args::SpikingArgs, offset::Real)
    #determine the ending time of each cycle
    cycles = generate_cycles(tspan, spk_args, offset)
    inds = [argmin(abs.(t .- t_c)) for t_c in cycles]

    #sample the potential at the end of each cycle
    u = u[inds] |> stack
    ts = t[inds]
    train = solution_to_train(u, ts, spk_args=spk_args, offset=offset)
    return train
end

"""
This implementation takes a single matrix at pre-selected, representative times and converts each temporal slice
to spikes.
"""
function solution_to_train(u::AbstractArray{<:Complex}, times::AbstractVector{<:Real}; spk_args::SpikingArgs, offset::Real)
    #determine the ending time of each cycle
    spiking = abs.(u) .> spk_args.threshold
    
    #convert the phase represented by that potential to a spike time
    tms = potential_to_time(u, times, spk_args = spk_args)
    
    if on_gpu(tms)
        gpu = true
        spiking = spiking |> cdev
        tms = tms |> cdev
    else
        gpu = false
    end

    #return only the times where the neuron is spiking
    cut_index = i -> CartesianIndex(Tuple(i)[1:end-1])
    inds = findall(spiking)
    tms = tms[inds]
    inds = cut_index.(inds)
    train = SpikeTrain(inds, tms, size(u)[1:end-1], offset + spiking_offset(spk_args))

    if gpu
        train = SpikeTrainGPU(train)
    end

    return train
end