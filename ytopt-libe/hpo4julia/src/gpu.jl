include("domains.jl")

const N_THREADS = 256

# Define devices
cdev = cpu_device()
if CUDA.functional()
    gdev = gpu_device()
end

function on_gpu(args...)
    locs = [typeof(x) <: CuArray for x in args]
    return reduce(+, locs) > 0
end

#Kernels 

function threads_blks(l::Int, threads::Int = N_THREADS)
    blocks = cld(l, threads)
    return threads, blocks
end

function gaussian_kernel_gpu(x::Float32, t::Float32, t_sigma::Float32)
    i = exp(-1.0f0 * ((t - x) / (2.0f0 * t_sigma))^2.0f0)
    return i
end

function scatter_add_kernel!(output, values, indices)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(indices)
        index = indices[i]
        value = values[i]
        CUDA.@atomic output[index] += value
    end
    return nothing
end

function parallel_scatter_add(indices::CuArray{Int}, values::CuArray{T}, output_size::Int) where T
    @assert length(indices) == length(values) "Length of indices and values must match"
    
    output = CUDA.zeros(T, output_size)
    threads = 256
    blocks = cld(length(indices), threads)
    
    @cuda threads=threads blocks=blocks scatter_add_kernel!(output, values, indices)
    
    return output
end

function interference_kernel(A, B, output, X, M, N, D)
    m = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    x = blockIdx().z

    if x <= X && m <= M && n <= N
        acc = 0.0f0
        @inbounds for d in 1:D
            a = A[d, m, x]
            b = B[d, n, x]
            
            sum_real = real(a) + real(b)
            sum_imag = imag(a) + imag(b)
            interference = sqrt(sum_real^2 + sum_imag^2)
            
            magnitude = clamp(interference, 0.0f0, 2.0f0)
            half_angle = acos(0.5f0 * magnitude)
            sim = cos(2.0f0 * half_angle)
            acc += sim
        end
        output[m, n, x] = acc / D
    end
    return
end

# Domains

function potential_to_phase(potential::CuArray, ts::AbstractVector; spk_args::SpikingArgs, offset::Real=0.0f0, threshold::Bool=false)
    @assert size(potential)[end] == length(ts) "Time dimensions must match"
    dims = collect(1:ndims(potential))

    #find the angle of a neuron representing 0 phase at the current moment in time
    current_zeros = cu(phase_to_potential.(0.0f0, ts, offset=offset, spk_args=spk_args))

    #get the arc subtended in the complex plane between that reference and our neuron potentials
    potential = permutedims(potential, reverse(dims))
    arc = angle.(current_zeros) .- angle.(potential)
    
    #normalize by pi and shift to -1, 1
    phase = mod.((arc ./ pi_f32 .+ 1.0f0), 2.0f0) .- 1.0f0

    #replace silent neurons with NaN
    silent = abs.(potential) .< spk_args.threshold
    phase[silent] .= Float32(NaN)
    phase = permutedims(phase, reverse(dims))
    
    return phase
end

function phase_to_train(phases::CuArray; spk_args::SpikingArgs, repeats::Int = 1, offset::Real = 0.0f0)
    shape = phases |> size
    indices = collect(CartesianIndices(shape)) |> vec
    times = phase_to_time(phases, spk_args=spk_args, offset=offset) |> vec

    if repeats > 1
        n_t = times |> length
        offsets = cu(repeat(collect(0:repeats-1) .* spk_args.t_period, inner=n_t))
        times = repeat(times, repeats) .+ offsets
        indices = repeat(indices, repeats)
    end

    train = SpikeTrainGPU(indices, times, shape, offset)
    return train
end

#Spiking

function parallel_current(stg::SpikeTrainGPU, t::Float32, spk_args::SpikingArgs)
    currents = gaussian_kernel_gpu.(stg.times, t, Float32(spk_args.t_window))
    output = parallel_scatter_add(stg.linear_indices, currents, stg.linear_shape)

    return output
end

function spike_current(train::SpikeTrainGPU, t::Float32, spk_args::SpikingArgs)
    scale = spk_args.spk_scale
    current = parallel_current(train, t, spk_args)
    current = reshape(current, train.shape)
    
    return current
end

function bias_current(bias::CuArray{<:Complex}, t::Real, t_offset::Real, spk_args::SpikingArgs)
    phase = complex_to_angle(bias)
    mag = abs.(bias)
    return bias_current(phase, mag, t, t_offset, spk_args)
end

function bias_current(phase::CuArray{<:Real}, mag::CuArray{<:Real}, t::Real, t_offset::Real, spk_args::SpikingArgs)
    #what times to the bias values correlate to?
    times = phase_to_time(phase, spk_args=spk_args, offset=t_offset)
    #determine the time within the cycle
    t = Float32(mod(t, spk_args.t_period))
    #add the active currents, scaled by the gaussian kernel & bias magnitude
    bias = mag .* gaussian_kernel_gpu.(times, t, Float32(spk_args.t_window))

    return bias
end

function f32_tspan(tspan::Tuple{<:Real, <:Real})
    tspan = (Float32(tspan[1]), Float32(tspan[2]))
    return tspan
end

function oscillator_bank(u0::CuArray, dzdt::Function; tspan::Tuple{<:Float32, <:Float32}, spk_args::SpikingArgs)
    #solve the memory compartment
    prob = ODEProblem(dzdt, u0, tspan)
    sol = solve(prob, spk_args.solver; spk_args.solver_args...)
    
    return sol
end

function oscillator_bank(x::SpikeTrainGPU; tspan::Tuple{<:Real, <:Real} = (0.0f0, 10.0f0), spk_args::SpikingArgs)
    tspan = tspan |> f32_tspan 
    update_fn = spk_args.update_fn

    #set up compartments for each sample
    u0 = CUDA.zeros(ComplexF32, x.shape)
    #resonate in time with the input spikes
    dzdt(u, p, t) = update_fn(u) .+ spike_current(x, t, spk_args)

    sol = oscillator_bank(u0, dzdt, tspan=tspan, spk_args=spk_args)

    return sol
end

function oscillator_bank(x::SpikeTrainGPU, w::AbstractMatrix, b::AbstractVecOrMat; kwargs...)
    return oscillator_bank(x, cu(w), cu(b); kwargs...)
end

function oscillator_bank(x::SpikeTrainGPU{2}, w::CuArray, b::CuArray; tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs)
    tspan = tspan |> f32_tspan
    #set up functions to define the neuron's differential equations
    update_fn = spk_args.update_fn
    #get the number of batches & output neurons
    output_shape = (size(w, 1), x.shape[2])
    u0 = CUDA.zeros(ComplexF32, output_shape)

    #solve the ODE over the given time span
    dzdt(u, p, t) = update_fn(u) + w * spike_current(x, t, spk_args) .+ bias_current(b, t, x.offset, spk_args)
    sol = oscillator_bank(u0, dzdt, tspan=tspan, spk_args=spk_args)

    #return full solution
    return sol
end

function oscillator_bank(x::SpikeTrainGPU{3}, w::CuArray, b::CuArray; tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs)
    tspan = tspan |> f32_tspan
    #set up functions to define the neuron's differential equations
    update_fn = spk_args.update_fn
    #get the number of batches & output neurons
    output_shape = (size(w, 1), x.shape[2], x.shape[3])
    u0 = CUDA.zeros(ComplexF32, output_shape)

    #solve the ODE over the given time span
    dzdt(u, p, t) = update_fn(u) + batched_mul(x, spike_current(x, t, spk_args)) .+ bias_current(b, t, x.offset, spk_args)
    sol = oscillator_bank(u0, dzdt, tspan=tspan, spk_args=spk_args)

    #return full solution
    return sol
end

function oscillator_bank(x::SpikeTrainGPU{4}, w::CuArray, b::CuArray; tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs)
    tspan = tspan |> f32_tspan
    #set up functions to define the neuron's differential equations
    update_fn = spk_args.update_fn
    #get the number of batches, channels, & output neurons
    output_shape = (size(w, 1), size(w,2), x.shape[2], x.shape[3])
    u0 = CUDA.zeros(ComplexF32, output_shape)

    #solve the ODE over the given time span
    dzdt(u, p, t) = update_fn(u) + batched_mul(x, spike_current(x, t, spk_args)) .+ bias_current(b, t, x.offset, spk_args)
    sol = oscillator_bank(u0, dzdt, tspan=tspan, spk_args=spk_args)

    #return full solution
    return sol
end

function similarity_outer(A::CuArray{ComplexF32,3}, B::CuArray{ComplexF32,3})
    # Vectorized implementation that avoids custom CUDA kernels and uses
    # broadcasting + reductions. This form is much simpler for AD backends
    # (Zygote) to trace through on GPU arrays.
    D, M, X = size(A)
    D_B, N, X_B = size(B)
    @assert X == X_B "Batch size mismatch (X_A=$X vs X_B=$X_B)"
    @assert D == D_B "Feature dimension mismatch (D_A=$D vs D_B=$D_B)"

    # Separate into real and imaginary parts
    realA = real.(A)  # (D, M, X)
    imagA = imag.(A)
    realB = real.(B)  # (D, N, X)
    imagB = imag.(B)

    # Reshape for broadcasting to (D, M, N, X)
    realA4 = reshape(realA, D, M, 1, X)
    realB4 = reshape(realB, D, 1, N, X)
    imagA4 = reshape(imagA, D, M, 1, X)
    imagB4 = reshape(imagB, D, 1, N, X)

    # Compute pairwise sums across vectors for each feature d and batch x
    sum_real = realA4 .+ realB4
    sum_imag = imagA4 .+ imagB4

    # Instead of sqrt -> acos -> cos, simplify using trig identity:
    # sim = cos(2 * acos(0.5 * mag)) == 0.5 * mag^2 - 1
    # where mag = clamp(|u+v|, 0, 2). So mag^2 = clamp(|u+v|^2, 0, 4).
    sq = sum_real .^ 2 .+ sum_imag .^ 2
    sq_clamped = clamp.(sq, 0.0f0, 4.0f0)
    sim_per_d = 0.5f0 .* sq_clamped .- 1.0f0  # (D, M, N, X)

    # Average over the D dimension
    sim_avg = mean(sim_per_d, dims=1)  # (1, M, N, X)
    sim_avg = dropdims(sim_avg, dims=1) # (M, N, X)

    # Match CPU implementation ordering: CPU stacks and then permutes to
    # (slice_y_index, batch_index, slice_x_index) using (2,3,1).
    # GPU current sim_avg is (M, N, X) where M corresponds to slices of A
    # and N to slices of B. Permute to match CPU's batch-last ordering.
    sim_avg = permutedims(sim_avg, (2,3,1)) # -> (N, X, M)

    return sim_avg
end

function similarity_outer(A::CuArray{ComplexF32,2}, B::CuArray{ComplexF32,2})
    # Treat 2D inputs as 3D with a singleton batch dimension and delegate
    # to the vectorized 3D implementation.
    D_A, M = size(A)
    D_B, N = size(B)
    @assert D_A == D_B "Feature dimension mismatch (D_A=$D_A vs D_B=$D_B)"

    A3 = reshape(A, D_A, M, 1)
    B3 = reshape(B, D_B, N, 1)

    out3 = similarity_outer(A3, B3)

    # After the 3D implementation permutation, out3 will have shape
    # (N, 1, M). Reshape to (N, M) to match the CPU 2D similarity_outer
    # which returns permuted result (dims swapped)
    return reshape(out3, size(out3,1), size(out3,3))
end

# Support dispatch when inputs are real-valued CuArrays by converting to
# ComplexF32 on the device and delegating to the complex-valued GPU kernels.
function similarity_outer(A::CuArray{<:Real,3}, B::CuArray{<:Real,3}; dims=2)
    # Convert phase angles to complex representation on GPU and delegate to
    # the complex-valued, vectorized implementation.
    A_c = angle_to_complex(A)
    B_c = angle_to_complex(B)
    return similarity_outer(A_c, B_c)
end

function similarity_outer(A::CuArray{<:Real,2}, B::CuArray{<:Real,2}; dims=2)
    A_c = angle_to_complex(A)
    B_c = angle_to_complex(B)
    return similarity_outer(A_c, B_c)
end