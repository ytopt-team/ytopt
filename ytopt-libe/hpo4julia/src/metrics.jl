include("network.jl")

#Misc. metrics

function arc_error(phase::Real)
    return sin(pi_f32 * phase)
end

function arc_error(phases::AbstractArray)
    return arc_error.(phases)
end

function angular_mean(phases::AbstractArray; dims)
    u = exp.(pi_f32 * 1.0f0im .* phases)
    u_mean = mean(u, dims=dims)
    phase = angle.(u_mean) ./ pi_f32
    return phase
end

function exp_score(similarity::AbstractArray; scale::Real = 3.0f0)
    return exp.((1.0f0 .- similarity) .* scale) .- 1.0f0
end

function z_score(phases::AbstractArray)
    arc = remap_phase(phases .- 0.5f0)
    score = abs.(atanh.(arc))
    return score
end

function similarity_correlation(static_similarity::Matrix{<:Real}, dynamic_similarity::Array{<:Real,3})
    n_steps = axes(dynamic_similarity, 3)
    cor_vals = [cor_realvals(static_similarity |> vec, dynamic_similarity[:,:,i] |> vec) for i in n_steps]
    return cor_vals
end

function cycle_correlation(static_phases::Matrix{<:Real}, dynamic_phases::Array{<:Real,3})
    n_cycles = axes(dynamic_phases, 3)
    cor_vals = [cor_realvals(static_phases |> vec, dynamic_phases[:,:,i] |> vec) for i in n_cycles]
    return cor_vals
end

function cycle_sparsity(static_phases::Matrix{<:Real}, dynamic_phases::Array{<:Real,3})
    n_cycles = axes(dynamic_phases, 3)
    total = reduce(*, size(static_phases))
    sparsity_vals = [sum(isnan.(dynamic_phases[:,:,i])) / total for i in n_cycles]
    return sparsity_vals
end

function cor_realvals(x, y)
    is_real = x -> .!isnan.(x)
    x_real = is_real(x)
    y_real = is_real(y)
    reals = x_real .* y_real
    if sum(reals) == 0
        return 0.0f0
    else
        return cor(x[reals], y[reals])
    end
end

function OvR_matrices(predictions, labels, threshold::Real)
    #get the confusion matrix for each class verus the rest
    mats = diag([confusion_matrix(ys, ts, threshold) for ys in eachslice(predictions, dims=1), ts in eachslice(labels, dims=1)])
    return mats
end

function tpr_fpr(prediction, labels, points::Int = 201, epsilon::Real = 0.01f0)
    test_points = range(start = 0.0f0, stop = -20.0f0, length = points)
    test_points = vcat(exp.(test_points), 0.0f0, reverse(-1.0f0 .* exp.(test_points)))
    fn = x -> sum(OvR_matrices(prediction, labels, x))
    confusion = cat(fn.(test_points)..., dims=3)

    classifications = dropdims(sum(confusion, dims=2), dims=2)
    cond_true = classifications[1,:]
    cond_false = classifications[2,:]

    #return cond_true, cond_false

    true_positives = confusion[1,1,:]
    false_positives = confusion[2,1,:]

    #return true_positives, false_positives

    tpr = true_positives ./ cond_true
    fpr = false_positives ./ cond_false

    return tpr, fpr
end
    
function interpolate_roc(roc)
    tpr, fpr = roc
    interp = linear_interpolation(fpr, tpr)
    return interp
end

function dense_onehot(x::OneHotMatrix)
    return 1.0f0 .* x
end

function confusion_matrix(sim, truth, threshold::Real)
    truth = hcat(truth .== 1, truth .== 0)
    prediction = hcat(sim .> threshold, sim .<= threshold)

    confusion = truth' * prediction
    return confusion
end


#Loss functions

function quadrature_loss(phases::AbstractArray, truth::AbstractArray; dim::Int = 1)
    targets = 0.5f0 .* truth
    sim = similarity(phases, targets, dim = dim)
    return 1.0f0 .- sim
end

function similarity_loss(similarities::AbstractArray, truth::AbstractArray; dim::Int = 1)
    distance = abs.(1.0 .- similarities) .* truth
    distance = sum(distance .* truth, dims = dim)
    loss = 2.0f0 .* sin.(pi_f32/4.0f0 .* distance) .^ 2.0f0
    return loss
end

function evaluate_loss(predictions::AbstractArray, truth::AbstractArray, encoding::Symbol = :similarity; reduce_dim::Int = 1)
    if encoding == :quadrature
        loss_fn = quadrature_loss
    else
        loss_fn = similarity_loss
    end

    #match the loss dispatch dimensions against the truth & predictions
    n_d_pred = ndims(predictions)
    n_d_truth = ndims(truth)
    if n_d_pred == 2 && n_d_truth == 2
        return loss_fn(predictions, truth, dim=reduce_dim)
    else
        dispatch_dims = setdiff(Set(1:n_d_pred), Set(1:n_d_truth)) |> Tuple
        return map(x -> loss_fn(x, truth, dim=reduce_dim), eachslice(predictions, dims=dispatch_dims))
    end
end

function evaluate_loss(predictions::SpikingCall, truth::AbstractArray, encoding::Symbol = :similarity; reduce_dim::Int = 1)
    predictions = train_to_phase(predictions) 
    truth = truth
    return evaluate_loss(predictions, truth, encoding, reduce_dim=reduce_dim)
end

# Prediction functions
function predict_quadrature(phases::AbstractArray; dim::Int=1)
    if on_gpu(phases)
        phases = phases |> cdev
    end

    predictions = getindex.(argmin(abs.(phases .- 0.5f0), dims=dim), dim)
    return predictions
end

function predict_similarity(sims::AbstractArray; dim::Int=1)
    if on_gpu(sims)
        sims = sims |> cdev
    end

    predictions = vec(getindex.(argmax(sims, dims=dim), dim))
    return predictions
end

function predict(predictions::AbstractArray, encoding::Symbol = :similarity; reduce_dim=1)
    if encoding == :quadrature
        predict_fn = x -> predict_quadrature(x, dim=reduce_dim)
    else
        predict_fn = x -> predict_similarity(x, dim=reduce_dim)
    end

    return predict_fn(predictions)
end

function predict(predictions::SpikingCall, encoding::Symbol = :similarity; reduce_dim::Int=1)
    predictions = train_to_phase(predictions)
    return predict(predictions, encoding, reduce_dim=reduce_dim)
end

# Performance evaluation functions
function evaluate_accuracy(values::AbstractArray, truth::AbstractArray, encoding::Symbol; reduce_dim::Int=1)
    if on_gpu(values, truth)
        values = values |> cdev
        truth = truth |> cdev
    end

    @assert ndims(values) >= ndims(truth) "Dimensionality of truth must be able to map onto values"
    reshape_dims = [d == reduce_dim ? 1 : size(truth,d) for d in 1:ndims(truth)]
    idx = reshape(getindex.(findall(truth .== 1.0f0), reduce_dim), reshape_dims...)
    predict_fn = x -> sum(predict(x, encoding, reduce_dim=reduce_dim) .== idx)
    n_truth = prod([size(truth, d) for d in setdiff(Set(1:ndims(truth)), Set(reduce_dim))])

    if ndims(values) > ndims(truth)
        dispatch_dims = setdiff(Set(1:ndims(values)), Set(1:ndims(truth))) |> Tuple
        response = map(predict_fn, eachslice(values, dims=dispatch_dims))
    else
        response = predict_fn(values)
    end
    
    return response, n_truth
end

function evaluate_accuracy(values::SpikingCall, truth::AbstractArray, encoding::Symbol; reduce_dim::Int=1)
    values = train_to_phase(values)
    return evaluate_accuracy(values, truth, encoding, reduce_dim=reduce_dim)
end

function loss_and_accuracy(data_loader, model, ps, st, args; reduce_dim::Int=1, encoding::Symbol = :codebook)
    loss_fn = (x, y) -> evaluate_loss(x, y, encoding, reduce_dim=reduce_dim)

    if args.use_cuda && CUDA.functional()
        dev = gdev
    else
        dev = cdev
    end

    num = 0
    correct = 0
    ls = 0.0f0

    for (x, y) in data_loader
        x = x |> dev
        y = y |> dev
        ŷ, _ = model(x, ps, st)
        @assert typeof(ŷ) != SpikingCall "Must call spiking models with SpikingArgs provided"

        ls += sum(stack(cdev(loss_fn(ŷ, y)))) #sum across batches
        model_correct, answers = cdev.(evaluate_accuracy(ŷ, y, encoding, reduce_dim=reduce_dim))
        correct += model_correct
        num += answers
    end

    return ls / num, correct / num
end

function spiking_loss_and_accuracy(data_loader, model, ps, st, args; reduce_dim::Int=1, encoding::Symbol = :codebook, repeats::Int)
    loss_fn = (x, y) -> evaluate_loss(x, y, encoding, reduce_dim=reduce_dim)

    if args.use_cuda && CUDA.functional()
        dev = gdev
    else
        dev = cdev
    end

    num = 0
    correct = zeros(Int64, repeats)
    ls = zeros(Float32, (1,repeats))

    for (x, y) in data_loader
        x = x |> dev
        y = y |> dev
        ŷ, _ = model(x, ps, st)
        ls .+= sum(stack(cdev(loss_fn(ŷ, y))), dims=1) #sum across batches
        model_correct, answers = cdev.(evaluate_accuracy(ŷ, y, encoding, reduce_dim=reduce_dim))
        correct .+= model_correct
        num += answers
    end

    return ls ./ num, correct ./ num
end