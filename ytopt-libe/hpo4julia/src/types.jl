include("imports.jl")

pi_f32 = convert(Float32, pi)

@kwdef mutable struct Args #lr is intentionally Float64 for Optimisers compatibility with some AD backends if not careful
    lr::Float64 = 0.0003       ## learning rate
    batchsize::Int = 128    ## batch size
    epochs::Int = 10        ## number of epochs
    use_cuda::Bool = true   ## use gpu (if cuda available)
    rng::Xoshiro = Xoshiro(42) ## global rng
end

struct SpikeTrain{N}
    indices::Array{<:Union{Int, CartesianIndex},1}
    times::Array{Float32,1}
    shape::Tuple
    offset::Float32

    function SpikeTrain(indices::AbstractArray,
        times::AbstractArray,
        shape::Tuple,
        offset::Real)
        times_f32 = eltype(times) == Float32 ? times : Float32.(times)
        return new{length(shape)}(indices,
                            times_f32,
                            shape,
                            Float32(offset))
    end
end


struct SpikeTrainGPU{N}
    indices::CuArray
    linear_indices::CuArray
    times::CuArray{<:Real}
    shape::Tuple
    linear_shape::Int
    offset::Float32

    function SpikeTrainGPU(indices::AbstractArray,
                            times::AbstractArray,
                            shape::Tuple,
                            offset::Real)
        times_f32 = eltype(times) == Float32 ? times : Float32.(times)
        cu_times_f32 = cu(times_f32)
        return new{length(shape)}(cu(indices),
                CuArray(LinearIndices(shape)[indices]),
                cu_times_f32,
                shape,
                reduce(*, shape),
                Float32(offset))
    end
end

function SpikeTrainGPU(st::SpikeTrain)
    return SpikeTrainGPU(st.indices,
                        st.times,
                        st.shape,
                        st.offset)
end

function SpikeTrain(stg::SpikeTrainGPU)
    st = SpikeTrain(Array(stg.indices),
                    Array(stg.times),
                    stg.shape,
                    stg.offset)
    return st
end

function Base.convert(::Type{SpikeTrain}, stg::SpikeTrainGPU)
    return SpikeTrain(stg)
end

function Base.convert(::Type{SpikeTrainGPU}, st::SpikeTrain)
    return SpikeTrainGPU(st)
end

SpikingTypes = Union{SpikeTrain, SpikeTrainGPU}
LuxParams = Union{NamedTuple, AbstractArray}

function Base.show(io::IO, train::SpikeTrain)
    print(io, "Spike Train: ", train.shape, " with ", length(train.times), " spikes.")
end

function Base.size(x::Union{SpikeTrain, SpikeTrainGPU})
    return x.shape
end

function remap(indices::Vector{<:CartesianIndex})
    #find the dimensions of unique elements for the new array
    n_dims = length(indices[1])
    elements = [map(x -> getindex(x, i), indices) for i in 1:n_dims]
    unique_elements = unique.(elements)
    new_ranges = [1:length(e) for e in unique_elements]
    #for each dimension and element, construct the map of old to new elements
    match = (x, i) -> x => new_ranges[i][findfirst(unique_elements[i] .== x)[1]]
    mapping = [Dict(match.(elements[i], i)) for i in 1:n_dims]

    #map each old index to the new index
    new_indices = map(idx -> CartesianIndex([mapping[i][idx[i]] for i in 1:n_dims]...), indices)
    new_shape = Tuple([r[end] for r in new_ranges])
    return new_indices, new_shape
end

function Base.getindex(x::SpikeTrain, inds...)
    #find the relevant entries
    idxs = CartesianIndices(size(x))
    selected = idxs[inds...]
    matches = [idx in selected for idx in x.indices]
    #downselect
    sel_indices = x.indices[matches]
    sel_times = x.times[matches]
    #map the old indicies to new values
    new_inds, new_shape = remap(sel_indices)
    new_train = SpikeTrain(new_inds, sel_times, new_shape, x.offset)

    return new_train
end

function increment_indices(indices::Vector{CartesianIndex{N}}, dim::Int, value::Int) where N
    # Check if the dimension is valid
    if dim < 1 || dim > N
        error("Invalid dimension. Must be between 1 and $N.")
    end

    # Increment the indices along the specified dimension
    return [CartesianIndex(ntuple(i -> i == dim ? idx[i] + value : idx[i], Val(N))) for idx in indices]
end

function Base.cat(x::SpikeTrain...; dim)
    n_trains = length(x)

    inds = x[1].indices |> deepcopy
    tms = x[1].times |> deepcopy
    offset = x[1].offset
    n_dims = length(x[1].shape)

    if n_trains > 1
        i = 2

        for i in 2:n_trains
            #calculate how far the indices need to be shifted
            selected = getindex.(inds, dim)
            idx_offset = maximum(selected)
            
            #offset the indices
            new_inds = increment_indices(x[i].indices, dim, idx_offset)
            #append them to the new train
            append!(inds, new_inds)
            append!(tms, x[i].times)
            @assert x[i].offset == offset "Spike trains must have idential offset to concatentate"
        end
    end

    shape = Tuple([maximum(getindex.(inds, i)) for i in 1:n_dims])

    #create and return the new train
    new_train = SpikeTrain(inds, tms, shape, offset)
end

function Base.size(x::SpikeTrain, d::Int)
    return x.shape[d]
end

function get_time(st::SpikeTrainGPU, times::Tuple{<:Real, <:Real})
    return get_time(SpikeTrain(st), times)
end

function get_time(st::SpikeTrain, times::Tuple{<:Real, <:Real})
    valid = findall((st.times .> times[1]) .* (st.times .< times[2]))
    st = SpikeTrain(st.indices[valid], st.times[valid], st.shape, st.offset)
    return st
end

struct SpikingArgs
    leakage::Float32
    t_period::Float32
    t_window::Float32
    spk_scale::Float32
    threshold::Float32
    spike_kernel::Union{Symbol, Function}
    solver # Solver type can be kept generic
    solver_args::Dict
    update_fn::Function
end

function SpikingArgs(; leakage::Real = -0.2f0, 
                    t_period::Real = 1.0f0,
                    t_window::Real = 0.01f0,
                    spk_scale::Real = 1.0f0,
                    threshold::Real = 0.001f0,
                    spike_kernel = :gaussian,
                    solver = Heun(),
                    solver_args = Dict(:dt => 0.01f0,
                                    :adaptive => false,
                                    :sensealg => InterpolatingAdjoint(; autojacvec=ZygoteVJP(allow_nothing=false)),
                                    :save_start => true))
                    
    return SpikingArgs(Float32(leakage),
            Float32(t_period),
            Float32(t_window),
            Float32(spk_scale),
            Float32(threshold),
            spike_kernel,
            solver,
            solver_args,
            u -> neuron_constant(Float32(leakage), Float32(t_period)) .* u,)
end

function SpikingArgs_NN(; leakage::Real = -0.2f0, # Changed default to Float32 literal
    t_period::Real = 1.0f0,
    t_window::Real = 0.01f0, # Ensure all Real args are consistently handled
    spk_scale::Real = 1.0f0,
    threshold::Real = 0.001f0,
    spike_kernel = :gaussian,
    solver = Heun(),
    solver_args = Dict(:dt => 0.01f0,
                    :adaptive => false,
                    :sensealg => InterpolatingAdjoint(; autojacvec=ZygoteVJP(allow_nothing=false)),
                    :save_start => true),
    update_fn::Function)

    return SpikingArgs(Float32(leakage),
            Float32(t_period),
            Float32(t_window),
            Float32(spk_scale),
            Float32(threshold),
            spike_kernel,
            solver,
            solver_args,
            update_fn)
end

function Base.show(io::IO, spk_args::SpikingArgs)
    print(io, "Neuron parameters: Period ", spk_args.t_period, " (s)\n")
    print(io, "Current kernel duration: ", spk_args.t_window, " (s)\n")
    print(io, "Threshold: ", spk_args.threshold, " (V)\n")
end

struct SpikingCall
    train::SpikingTypes
    spk_args::SpikingArgs
    t_span::Tuple{Float32, Float32}
end

function n_cycles(call::SpikingCall)
    return Int(floor(call.t_span[2] / call.spk_args.t_period))
end

function Base.size(x::SpikingCall)
    return x.train.shape
end

function Base.getindex(x::SpikingCall, inds...)
    new_train = getindex(x.train, inds...)
    new_call = SpikingCall(new_train, x.spk_args, x.t_span)
    return new_call
end

struct LocalCurrent
    current_fn::Function
    shape::Tuple
    offset::Float32
end

function LocalCurrent(current_fn::Function, shape::Tuple, offset::Real)
    return LocalCurrent(current_fn, shape, Float32(offset))
end

function LocalCurrent(current_fn::Function, shape::Tuple) # Default offset
    return LocalCurrent(current_fn, shape, 0.0f0)
end

function LocalCurrent(st::SpikingTypes, spk_args::SpikingArgs)
    return LocalCurrent(t -> spike_current(st, t, spk_args),
                                        st.shape,
                                        st.offset)
end

function Base.size(x::LocalCurrent)
    return x.shape
end

struct CurrentCall
    current::LocalCurrent
    spk_args::SpikingArgs
    t_span::Tuple{<:Real, <:Real}
end

function CurrentCall(sc::SpikingCall)
    return CurrentCall(LocalCurrent(sc.train, sc.spk_args), 
                        sc.spk_args,
                        sc.t_span)
end

function Base.size(x::CurrentCall)
    return x.current.shape
end

PhaseInput = Union{SpikeTrain, SpikingCall, LocalCurrent, CurrentCall, AbstractArray, ODESolution}