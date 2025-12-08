include("vsa.jl")

"""
MakeSpiking - a layer to include in Chains to convert phase tensors
into SpikeTrains
"""
struct MakeSpiking <: Lux.AbstractLuxLayer
    spk_args::SpikingArgs
    repeats::Int
    tspan::Tuple{<:Real, <:Real}
    offset::Real
end

function MakeSpiking(spk_args::SpikingArgs, repeats::Int)
    return MakeSpiking(spk_args, repeats, (0.0f0, spk_args.t_period * repeats), 0.0f0)
end

function (a::MakeSpiking)(x::AbstractArray, params::LuxParams, state::NamedTuple)
    train = phase_to_train(x, spk_args = a.spk_args, repeats = a.repeats, offset = a.offset)
    call = SpikingCall(train, a.spk_args, a.tspan)
    return call, state
end

function (a::MakeSpiking)(x::ODESolution, params::LuxParams, state::NamedTuple)
    train = solution_to_train(x, a.tspan, spk_args = a.spk_args, offset = 0.0f0)
    call = SpikingCall(train, a.spk_args, a.tspan)
    return call, state
end

# Extend Lux.Flatten for SpikeTrain and SpikingCall types, preserving the last dimension (batch)
function (f::Lux.FlattenLayer)(x::SpikeTrain, params::LuxParams, state::NamedTuple)
    original_shape = x.shape
    N_dims = length(original_shape)

    if N_dims == 0
        # Cannot meaningfully flatten a 0-dimensional SpikeTrain for batch processing
        return x, state
    end

    local new_shape::Tuple
    local feature_shape_tuple::Tuple

    if N_dims == 1 # Input is (Batch,), new shape will be (1, Batch)
        batch_size = original_shape[1]
        new_shape = (1, batch_size)
        feature_shape_tuple = () # No feature dimensions to flatten
    else # Input is (F1, ..., Fk, Batch), new shape (F1*...*Fk, Batch)
        batch_size = original_shape[end]
        feature_shape_tuple = original_shape[1:end-1]
        num_features_flat = prod(feature_shape_tuple)
        new_shape = (num_features_flat, batch_size)
    end

    # Convert original indices to new CartesianIndices for the new_shape
    new_indices = map(x.indices) do original_idx_val
        # Ensure original_idx_val is CartesianIndex for original_shape
        ci_orig::CartesianIndex = original_idx_val isa CartesianIndex ? 
                                  original_idx_val : 
                                  CartesianIndices(original_shape)[original_idx_val]
        
        # Batch index is the component corresponding to the last dimension of original_shape
        batch_val = ci_orig[N_dims]

        # Extract feature coordinates as a tuple of integers
        # For N_dims=1, (N_dims - 1) is 0, so ntuple returns ()
        feature_coords_as_tuple = ntuple(d -> ci_orig[d], N_dims - 1)
        
        # Calculate linear index for the (potentially) flattened features
        # If feature_shape_tuple is empty (e.g. original was (Batch,)), linear_feature_idx is 1.
        linear_feature_idx = isempty(feature_shape_tuple) ? 
                             1 : 
                             LinearIndices(feature_shape_tuple)[feature_coords_as_tuple...]
        
        CartesianIndex(linear_feature_idx, batch_val)
    end

    flattened_train = SpikeTrain(new_indices, x.times, new_shape, x.offset)
    return flattened_train, state
end

function (f::Lux.FlattenLayer)(x::SpikeTrainGPU, params::LuxParams, state::NamedTuple)
    original_shape = x.shape
    N_dims = length(original_shape)

    if N_dims == 0
        return x, state
    end

    local new_shape::Tuple
    local feature_shape_tuple::Tuple

    if N_dims == 1 # Input is (Batch,), new shape will be (1, Batch)
        batch_size = original_shape[1]
        new_shape = (1, batch_size)
        feature_shape_tuple = ()
    else # Input is (F1, ..., Fk, Batch), new shape (F1*...*Fk, Batch)
        batch_size = original_shape[end]
        feature_shape_tuple = original_shape[1:end-1]
        num_features_flat = prod(feature_shape_tuple)
        new_shape = (num_features_flat, batch_size)
    end

    # Convert original linear_indices (for N-D shape) to new CartesianIndices (for 2D shape) on CPU
    linear_indices_cpu = Array(x.linear_indices)
    cart_indices_obj_orig = CartesianIndices(original_shape)
    original_cart_indices_cpu = map(li -> cart_indices_obj_orig[li], linear_indices_cpu)

    new_cart_indices_cpu = map(original_cart_indices_cpu) do ci_orig
        batch_val = ci_orig[N_dims]
        feature_coords_as_tuple = ntuple(d -> ci_orig[d], N_dims - 1)
        linear_feature_idx = isempty(feature_shape_tuple) ? 1 : LinearIndices(feature_shape_tuple)[feature_coords_as_tuple...]
        CartesianIndex(linear_feature_idx, batch_val)
    end
    
    # SpikeTrainGPU constructor takes AbstractArray for indices (here, CPU Array of CartesianIndex),
    # and will convert it to CuArray and compute new linear_indices for the new_shape.
    # x.times is already a CuArray.
    flattened_train = SpikeTrainGPU(new_cart_indices_cpu, x.times, new_shape, x.offset)
    return flattened_train, state
end

function (f::Lux.FlattenLayer)(call::SpikingCall, params::LuxParams, state::NamedTuple)
    flattened_train, _ = f(call.train, params, state) # Dispatch to SpikeTrain or SpikeTrainGPU method
    new_call = SpikingCall(flattened_train, call.spk_args, call.t_span)
    return new_call, state
end

"""
Extension of dropout to SpikeTrains
"""
function dropout(rng::AbstractRNG, x::SpikingCall, p::T, training, invp::T, dims) where {T}
    train = x.train
    n_s = length(train.indices)

    keep_inds = rand(rng, Float32, (n_s)) .>= p
    new_inds = train.indices[keep_inds]
    new_tms = train.times[keep_inds]
    new_train = SpikeTrain(new_inds, new_tms, train.shape, train.offset)
    new_call = SpikingCall(new_train, x.spk_args, x.t_span)

    return new_call, (), rng
end



"""
ComplexBias layer for use in Phase Networks - adds bias in the complex plane
    during calls
"""
struct ComplexBias <: LuxCore.AbstractLuxLayer
    dims
    init_bias
end

function default_bias(rng::AbstractRNG, dims::Tuple{Vararg{Int}})
    return ones(ComplexF32, dims)
end

function zero_bias(rng::AbstractRNG, dims::Tuple{Vararg{Int}})
    return zeros(ComplexF32, dims)
end

function ComplexBias(dims::Tuple{Vararg{Int}}; init_bias = default_bias)
    if init_bias == nothing
        init_bias = (rng, dims) -> zeros(ComplexF32, dims)
    end

    return ComplexBias(dims, init_bias)
end

function Base.show(io::IO, b::ComplexBias)
    print(io, "ComplexBias($(b.dims))")
end

function (b::ComplexBias)(x::AbstractArray{<:Complex}, params::LuxParams, state::NamedTuple)
    bias_val = params.bias_real .+ 1.0f0im .* params.bias_imag
    return x .+ bias_val, state 
end

function (b::ComplexBias)(params::LuxParams, state::NamedTuple; offset::Real = 0.0f0, spk_args::SpikingArgs)
    bias_val = params.bias_real .+ 1.0f0im .* params.bias_imag
    return t -> bias_current(bias_val, t, offset, spk_args)
end

function Lux.initialparameters(rng::AbstractRNG, bias::ComplexBias)
    bias = bias.init_bias(rng, bias.dims)
    bias_real = real.(bias)
    bias_imag = imag.(bias)
    params = (bias_real = bias_real, bias_imag = bias_imag)
    return params
end

function Lux.initialstates(rng::AbstractRNG, bias::ComplexBias)
    # ComplexBias is stateless by itself, but Lux convention is to return an empty NamedTuple
    # if no specific state is needed.
    return NamedTuple()
end

"""
PhasorDense layer - implementationof fundamental dense layer using phase tensors
"""
struct PhasorDense <: LuxCore.AbstractLuxContainerLayer{(:layer, :bias)}
    layer # the conventional layer used to transform inputs
    bias # the bias in the complex domain used to shift away from the origin
    activation # the activation function which converts complex values to real phases
    use_bias::Bool # apply the layer with the bias if true
    return_solution::Bool # return the full ODE solution from a spiking input
end

function PhasorDense(shape::Pair{<:Integer,<:Integer}, activation = identity; return_solution::Bool = false, init_bias = default_bias, use_bias::Bool = true, kwargs...)
    layer = Dense(shape, identity; use_bias=false, kwargs...)
    bias = ComplexBias((shape[2],); init_bias = init_bias)
    return PhasorDense(layer, bias, activation, use_bias, return_solution)
end

function Lux.initialstates(rng::AbstractRNG, l::PhasorDense)
    st_layer = Lux.initialstates(rng, l.layer)
    st_bias = Lux.initialstates(rng, l.bias)
    return (layer = st_layer, bias = st_bias)
end

# Calls
function (a::PhasorDense)(x::AbstractArray, params::LuxParams, state::NamedTuple)
    xz = angle_to_complex(x)
    #stateless calls to dense
    y_real, _ = a.layer(real.(xz), params.layer, state.layer)
    y_imag, _ = a.layer(imag.(xz), params.layer, state.layer)
    y = y_real .+ 1.0f0im .* y_imag

    if a.use_bias
        y_biased, st_updated_bias = a.bias(y, params.bias, state.bias)
        y_activated = a.activation(y_biased)
    else
        #passthrough
        st_updated_bias = state.bias
        y_activated = a.activation(y)
    end

    # New state for PhasorDense layer
    st_new = (dense = state.layer, bias = st_updated_bias)
    return y_activated, st_new
end

function (a::PhasorDense)(x::SpikingCall, params::LuxParams, state::NamedTuple)
    current_call = CurrentCall(x)
    return a(current_call, params, state)
end

function (a::PhasorDense)(x::CurrentCall, params::LuxParams, state::NamedTuple)
    #pass the params and dense kernel to the solver
    sol = oscillator_bank(x.current, a, params, state, tspan=x.t_span, spk_args=x.spk_args, use_bias=a.use_bias)
    if a.return_solution
        u = sol.u
        t = sol.t
        return (u, t), state
    end

    train = solution_to_train(sol, x.t_span, spk_args=x.spk_args, offset=x.current.offset)
    next_call = SpikingCall(train, x.spk_args, x.t_span)
    return next_call, state
end

###
### Convolutional Phasor Layer
###

struct PhasorConv <: LuxCore.AbstractLuxContainerLayer{(:layer, :bias)}
    layer
    bias
    activation
    use_bias::Bool
    return_solution::Bool
end

function PhasorConv(k::Tuple{Vararg{Integer}}, chs::Pair{<:Integer,<:Integer}, activation = identity; return_solution::Bool = false, init_bias = default_bias, use_bias::Bool = true, kwargs...)
    #construct the convolutional layer
    layer = Conv(k, chs, identity; use_bias=false, kwargs...)
    bias = ComplexBias(([1 for _ in 1:length(k)]...,chs[2],), init_bias = init_bias)
    return PhasorConv(layer, bias, activation, use_bias, return_solution)
end

function Lux.initialstates(rng::AbstractRNG, l::PhasorConv)
    st_layer = Lux.initialstates(rng, l.layer)
    st_bias = Lux.initialstates(rng, l.bias)
    return (layer = st_layer, bias = st_bias)
end

function (pc::PhasorConv)(x::AbstractArray, ps::LuxParams, st::NamedTuple)
    x = angle_to_complex(x)
    x_real = real.(x)
    x_imag = imag.(x)

    y_real_conv, _ = pc.layer(x_real, ps.layer, st.layer)
    y_imag_conv, _ = pc.layer(x_imag, ps.layer, st.layer)
    y = y_real_conv .+ 1.0f0im .* y_imag_conv

    # Apply bias
    if pc.use_bias
        y_biased, st_updated_bias = pc.bias(y, ps.bias, st.bias)
        y_activated = pc.activation(y_biased)
    else
        #passthrough
        st_updated_bias = st.bias
        y_activated = a.activation(y)
    end

    st_new = (layer = st.layer, bias = st_updated_bias)
    return y_activated, st_new
end

function (a::PhasorConv)(x::SpikingCall, params::LuxParams, state::NamedTuple)
    current_call = CurrentCall(x)
    return a(current_call, params, state)
end

function (a::PhasorConv)(x::CurrentCall, params::LuxParams, state::NamedTuple)
    #pass the params and dense kernel to the solver
    sol = oscillator_bank(x.current, a, params, state, tspan=x.t_span, spk_args=x.spk_args, use_bias=a.use_bias)
    if a.return_solution
        u = sol.u
        t = sol.t
        return (u, t), state
    end

    train = solution_to_train(sol, x.t_span, spk_args=x.spk_args, offset=x.current.offset)
    next_call = SpikingCall(train, x.spk_args, x.t_span)
    return next_call, state
end

###
### Codebook layer - converts a vector to a value of similarities
###

struct Codebook <: LuxCore.AbstractLuxLayer
    dims
    Codebook(x::Pair{<:Int, <:Int}) = new(x)
end

function Base.show(io::IO, cb::Codebook)
    print(io, "Codebook($(cb.dims))")
end

function Lux.initialparameters(rng::AbstractRNG, cb::Codebook)
    return NamedTuple()
end

function Lux.initialstates(rng::AbstractRNG, cb::Codebook)
    state = (codes = random_symbols(rng, (cb.dims[1], cb.dims[2])),)
    return state
end

function (cb::Codebook)(x::AbstractArray{<:Real}, params::LuxParams, state::NamedTuple)
    return similarity_outer(x, state.codes), state
end

function (cb::Codebook)(x::SpikingCall, params::LuxParams, state::NamedTuple)
    current_call = CurrentCall(x)
    return cb(current_call, params, state)
end

function (cb::Codebook)(x::CurrentCall, params::LuxParams, state::NamedTuple)
    code_currents = phase_to_current(state.codes, spk_args=x.spk_args, offset=x.current.offset, tspan=x.t_span)
    return similarity_outer(x, code_currents), state
end


###
### Layer which resonates with incoming input currents - mainly with one input and weakly with others
###
struct PhasorResonant <: Lux.AbstractLuxLayer
    shape::Int
    layer
    init_weight
    return_solution::Bool
    static::Bool
end

function PhasorResonant(n::Int, spk_args::SpikingArgs, return_solution::Bool = true, static::Bool = true)
    if static
        init_w = () -> Matrix(ones(Float32, 1) .* I(n))
    else
        init_w = rng -> square_variance(rng, n)
    end
        
    return PhasorResonant(n, Dense(n => n), init_w, return_solution, static)
end

function Lux.initialparameters(rng::AbstractRNG, layer::PhasorResonant)
    if layer.static
        params = NamedTuple()
    else
        params = (weight = layer.init_weight(rng))
    end
end

# Calls

function (a::PhasorResonant)(x::CurrentCall, params::LuxParams, state::NamedTuple)
    if a.static
        y = oscillator_bank(x.current, a.init_weight(), zeros(ComplexF32, (a.shape)), spk_args=x.spk_args, tspan = x.t_span, return_solution = a.return_solution)
    else    
        y = oscillator_bank(x.current, params, spk_args=x.spk_args, tspan = x.t_span, return_solution = a.return_solution)
    end

    return y, state
end

function (a::PhasorResonant)(x::SpikingCall, params::LuxParams)
    if a.static
        y = v_bundle_project(x, a.init_weight(), zeros(ComplexF32, (a.shape)), return_solution = a.return_solution)
    else
        y = v_bundle_project(x, params, spk_args = x.spk_args, return_solution = a.return_solution)
    end

    return y, state
end

###
### Random Projection Layer
###

struct RandomProjection <: Lux.AbstractLuxLayer
    dim::Int # The dimension being projected, typically the feature dimension
end

function Lux.initialparameters(rng::AbstractRNG, layer::RandomProjection)
    return NamedTuple() # No trainable parameters
end

function Lux.initialstates(rng::AbstractRNG, layer::RandomProjection)
    # Create a random projection matrix W of size (dim, dim).
    # This matrix will project a vector of length `dim` to another vector of length `dim`.
    # Stored in state as it's non-trainable.
    projection_weights = randn(rng, Float32, layer.dim, layer.dim)
    return (weights = projection_weights, rng = Lux.replicate(rng))
end

# Call
function (rp::RandomProjection)(x::AbstractArray, params::LuxParams, state::NamedTuple)
    # x is expected to have its first dimension match rp.dim
    # e.g., x can be (dim, batch_size) or (dim, H, W, batch_size)
    
    current_size = size(x)
    if current_size[1] != rp.dim
        error("Input first dimension $(current_size[1]) must match layer dimension $(rp.dim)")
    end

    local y::AbstractArray
    if ndims(x) == 1 # Input is a vector (dim,)
        y = state.weights * x
    else # Input is a batched tensor (dim, other_dims...)
        x_reshaped = reshape(x, rp.dim, :)
        y_reshaped = state.weights * x_reshaped
        y = reshape(y_reshaped, current_size)
    end
    
    return y, state # State is not modified in the forward pass for this layer
end

struct RandomPhaseProjection <: LuxCore.AbstractLuxLayer
    dims
end

function RandomPhaseProjection(dims::Tuple{Vararg{Int}})
    return RandomPhaseProjection(dims)
end

function Lux.initialparameters(rng::AbstractRNG, layer::RandomPhaseProjection)
    return NamedTuple() # No trainable parameters
end

function Lux.initialstates(rng::AbstractRNG, layer::RandomPhaseProjection)
    # Create a random projection matrix W of size (dim, dim).
    # This matrix will project a vector of length `dim` to another vector of length `dim`.
    # Stored in state as it's non-trainable.
    projection_weights = rand(rng, (-1.0f0, 1.0f0), layer.dims)
    return (weights = projection_weights, rng = Lux.replicate(rng))
end

function (p::RandomPhaseProjection)(x::AbstractArray, params::LuxParams, state::NamedTuple)
    y = batched_mul(x, state.weights)
    return y, state
end

"""
Residual blocks
"""

struct ResidualBlock <: LuxCore.AbstractLuxContainerLayer{(:ff,)}
    ff
end

function ResidualBlock(dimensions::Tuple{Vararg{Int}};)
    @assert length(dimensions) >= 2 "Must have at least 1 layer"
    #construct a Phasor MLP based on the given dimensions
    pairs = [dimensions[i] => dimensions[i+1] for i in 1:length(dimensions) - 1]
    layers = [PhasorDense(pair) for pair in pairs]
    ff = Chain(layers...)

    return ResidualBlock(ff)
end

function (rb::ResidualBlock)(x, ps, st)
    # MLP path
    ff_out, st_ff = rb.ff(x, ps.ff, st.ff)
    x = v_bind(x, ff_out)
    
    return x, st_ff
end

"""
Phasor QKV Attention
"""

function attend(q::AbstractArray{<:Real, 3}, k::AbstractArray{<:Real, 3}, v::AbstractArray{<:Real, 3}; scale::AbstractArray=[1.0f0,])
    #compute qk scores
    #produces (qt kt b)
    d_k = size(k,2)
    scores = exp.(scale .* similarity_outer(q, k, dims=2)) ./ d_k
    #do complex-domain matrix multiply of values by scores (kt v b)
    v = angle_to_complex(v)
    #multiply each value by the scores across batch
    #(v kt b) * (kt qt b) ... (v kt) * (kt qt) over b
    output = batched_mul(v, scores)
    output = complex_to_angle(output)
    return output, scores
end

function score_scale(potential::CuArray{<:Complex,3}, scores::CuArray{<:Real,3}; d_k::Int, scale::AbstractArray=[1.0f0,])
    @assert size(potential, 3) == size(scores,3) "Batch dimensions of inputs must match"

    scores = permutedims(scores, (2,1,3))
    d_k = size(scores,1)
    scores = exp.(scale .* scores) ./ d_k
    scaled = batched_mul(potential, scores)
    return scaled
end

function attend(q::SpikingTypes, k::SpikingTypes, v::SpikingTypes; spk_args::SpikingArgs, tspan::Tuple{<:Real, <:Real}=(0.0f0, 10.0f0), return_solution::Bool = false, scale::AbstractArray=[1.0f0,])
    #compute the similarity between the spike trains
    #produces [time][b qt kt]
    scores = similarity_outer(q, k, spk_args=spk_args, tspan=tspan)
    #convert the values to potentials
    d_k = size(k)[2]
    values = oscillator_bank(v, tspan=tspan, spk_args=spk_args)
    #multiply by the scores found at each time step
    output_u = score_scale.(values.u, scores, scale=scale, d_k=d_k)
    if return_solution 
        return output_u 
    end

    output = solution_to_train(output_u, values.t, tspan, spk_args=spk_args, offset=v.offset)
    return output, scores
end

struct PhasorAttention <: Lux.AbstractLuxLayer
    init_scale::Real
end

function PhasorAttention()
    return PhasorAttention(1.0f0)
end

function Lux.initialparameters(rng::AbstractRNG, attention::PhasorAttention)
    params = (scale = [attention.init_scale,],)
end

function (a::PhasorAttention)(q::AbstractArray, k::AbstractArray, v::AbstractArray, ps::LuxParams, st::NamedTuple)
    result, scores = attend(q, k, v, scale=ps.scale)

    return result, (scores=scores,)
end

identity_layer = Chain(x -> x,)

struct SingleHeadAttention <: LuxCore.AbstractLuxContainerLayer{(:q_proj, :k_proj, :v_proj, :attention, :out_proj)}
    q_proj
    k_proj
    v_proj
    attention
    out_proj
end

function SingleHeadAttention(d_input::Int, d_model::Int; init=variance_scaling, kwargs...)
    default_model = () -> Chain(ResidualBlock((d_input, d_model)))

    q_proj = get(kwargs, :q_proj, default_model())
    k_proj = get(kwargs, :k_proj, default_model())
    v_proj = get(kwargs, :v_proj, default_model())
    scale = get(kwargs, :scale, 1.0f0)
    attention = get(kwargs, :attention, PhasorAttention(scale))
    out_proj = get(kwargs, :out_proj, PhasorDense(d_model => d_input; init))
    

    SingleHeadAttention(
        q_proj,  # Query
        k_proj,  # Key
        v_proj,  # Value
        attention, # Attention mechanism
        out_proj,   # Output
    )
end

function (m::SingleHeadAttention)(q, kv, ps, st)
    q, _ = m.q_proj(q, ps.q_proj, st.q_proj)
    k, _ = m.k_proj(kv, ps.k_proj, st.k_proj)
    v, _ = m.v_proj(kv, ps.v_proj, st.v_proj)
    
    # Single-head attention (nheads=1)
    attn_out, scores = m.attention(q, k, v, ps.attention, st.attention)
    output = m.out_proj(attn_out, ps.out_proj, st.out_proj)[1]
    
    return output, (scores = scores,)
end

struct SingleHeadCABlock <: LuxCore.AbstractLuxContainerLayer{(:attn, :q_norm, :kv_norm, :ff_norm, :ff)}
    attn::SingleHeadAttention
    q_norm
    kv_norm
    ff_norm
    ff
end

function SingleHeadCABlock(d_input::Int, d_model::Int, n_q::Int, n_kv::Int; dropout::Real=0.1f0, kwargs...)
    SingleHeadCABlock(
        SingleHeadAttention(d_input, d_model; kwargs...),
        LayerNorm((d_model, n_q)),
        LayerNorm((d_model, n_kv)),
        LayerNorm((d_model, n_q)),
        Chain(PhasorDense(d_input => d_model),
            Dropout(dropout),
            PhasorDense(d_model => d_input)),
    )
end

function (tb::SingleHeadCABlock)(q, kv, mask, ps, st)
    # Attention path
    norm_q = tb.q_norm(q, ps.q_norm, st.q_norm)[1]
    norm_kv = tb.kv_norm(kv, ps.kv_norm, st.kv_norm)[1]
    attn_out, st_attn = tb.attn(q, kv, ps.attn, st.attn)
    x = v_bind(q, attn_out)
    
    # Feed-forward path
    norm_x = tb.ff_norm(x, ps.ff_norm, st.ff_norm)[1]
    ff_out, st_ff = tb.ff(x, ps.ff, st.ff)
    x = v_bind(x, ff_out)
    
    return x, merge(st_attn, st_ff)
end

function train(model, ps, st, train_loader, loss, args; optimiser = Optimisers.Adam, verbose::Bool = false, sample_gradients::Int = 0)
    if CUDA.functional() && args.use_cuda
       @info "Training on CUDA GPU"
       #CUDA.allowscalar(false)
       device = gpu_device()
   else
       @info "Training on CPU"
       device = cpu_device()
   end

   ## Optimizer
   opt_state = Optimisers.setup(optimiser(args.lr), ps)
   losses = []
   gradients = []
   step_count = 0

   ## Training
   for epoch in 1:args.epochs
       for (x, y) in train_loader
           step_count += 1
           x = x |> device
           y = y |> device
           
           lf = p -> loss(x, y, model, p, st)
           lossval, gs = withgradient(lf, ps)
           if verbose
               println(reduce(*, ["Epoch ", string(epoch), " loss: ", string(lossval)]))
           end
           append!(losses, lossval)
           
           # Save gradients if sampling is enabled and we're at a sampling step
           if sample_gradients > 0 && (step_count % sample_gradients == 0)
               push!(gradients, deepcopy(gs[1]))
           end
           
           opt_state, ps = Optimisers.update(opt_state, ps, gs[1]) ## update parameters
       end
   end
   
   if sample_gradients > 0
    return losses, ps, st, gradients
   else
    return losses, ps, st
   end
end


"""
Other utilities
"""

struct MinPool <: LuxCore.AbstractLuxWrapperLayer{:pool}
    pool
end

function MinPool()
    return MinPool(MaxPool())
end

function (mp::MinPool)(x, ps, st)
    y = -1.0f0 .* mp(-1.0f0 .* x, ps, st)
    return y
end 

struct TrackOutput{L<:Lux.AbstractLuxLayer} <: Lux.AbstractLuxLayer
    layer::L
end

# Forward parameter initialization to inner layer
Lux.initialparameters(rng::AbstractRNG, t::TrackOutput) = 
    (layer=Lux.initialparameters(rng, t.layer),)

# Forward state initialization and add output tracking
function Lux.initialstates(rng::AbstractRNG, t::TrackOutput)
    st_layer = Lux.initialstates(rng, t.layer)
    return merge(st_layer, (outputs=(),))
end

function (t::TrackOutput)(x, ps, st)
    y, st_layer = Lux.apply(t.layer, x, ps.layer, st)
    new_st = merge(st_layer, (outputs=(st.outputs..., y),))
    return y, new_st
end

function variance_scaling(rng::AbstractRNG, shape::Integer...; mode::String = "avg", scale::Real = 0.66f0)
    fan_in = shape[end]
    fan_out = shape[1]

    if mode == "fan_in"
        scale /= max(1.0f0, fan_in)
    elseif mode == "fan_out"
        scale /= max(1.0f0, fan_out)
    else
        scale /= max(1.0f0, (fan_in + fan_out) / 2.0f0)
    end

    stddev = sqrt(scale) / 0.87962566103423978f0
    return truncated_normal(rng, shape..., mean = 0.0f0, std = stddev)
end

function square_variance(rng::AbstractRNG, shape::Integer; kwargs...)
    weights = variance_scaling(rng, shape, shape; kwargs...)
    weights[diagind(weights)] .= 1.0f0
    return weights
end