# Activate the project environment
using Pkg
Pkg.activate("..")

# Include necessary source files and import packages
include("../src/PhasorNetworks.jl")
using .PhasorNetworks
using Lux, MLUtils, MLDatasets, OneHotArrays, Statistics, LuxCUDA
using Random: Xoshiro, AbstractRNG
using Base: @kwdef
using Zygote: withgradient
using Optimisers, ComponentArrays
using Statistics: mean
using DifferentialEquations: Heun, Tsit5
using JLD2

solver_args = Dict(:adaptive => false, :dt => 0.01f0)
spk_args = SpikingArgs(threshold = 0.001f0,
                    solver=Tsit5(),
                    solver_args = solver_args)

@kwdef mutable struct ExpArgs
    lr::Float64 = 3e-4       ## learning rate
    r_lo::Float32 = 0.1f0  ## soft angle lower bound
    r_hi::Float32 = 0.2f0  ## soft angle upper bound
    rng::Xoshiro = Xoshiro(42) ## global rng
    test_spiking::Bool = false
end

function train_and_test_conv(ea::ExpArgs)
    # --- Argument/Parameter Setup ---
    println("Setting up parameters...")
    args = Args(lr = ea.lr, batchsize = 128, epochs = 10, use_cuda = true) # As per notebook

    # Device setup
    device_fn = args.use_cuda ? gpu_device() : cpu_device()
    cdev = cpu_device() # for operations that need to be on CPU like onecold
    gdev = gpu_device() # for CUDA operations

    println(args.use_cuda ? "Training on CUDA GPU" : "Training on CPU")

    # --- Data Loading ---
    println("Loading FashionMNIST dataset...")
    train_data = MLDatasets.FashionMNIST(split=:train)
    test_data = MLDatasets.FashionMNIST(split=:test)

    train_loader = DataLoader(train_data, batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=args.batchsize)

    # --- Conventional Network ---
    println("\n--- Conventional Network ---")
    conv_model = Chain(LayerNorm((28, 28)),
                x -> reshape(x, (28, 28, 1, :)),
                Conv((16, 16), 1 => 3, relu),
                Conv((8, 8), 3 => 1, relu),
                FlattenLayer(),
                Dense(36 => 128, relu),
                Dense(128 => 10),
                softmax)

    ps_conv, st_conv = Lux.setup(args.rng, conv_model) .|> device_fn
    
    # Loss function for conventional network
    function conv_loss_function(x, y, model, ps, st)
        y_pred, _ = Lux.apply(model, x, ps, st)
        y_onehot = onehotbatch(y, 0:9)
        return CrossEntropyLoss(;logits=false, dims=1)(y_pred, y_onehot)
    end

    function test_conv(model, data_loader, ps, st)
        # Evaluation phase
        total_correct = 0
        total_samples = 0
        for (x, y) in data_loader
            x = x |> device_fn
            
            y_pred, _ = Lux.apply(model, x, ps, st)
            pred_labels = onecold(cdev(y_pred))
            
            total_correct += sum(pred_labels .== y .+ 1)
            total_samples += length(y)
        end

        acc = total_correct / total_samples
    end

    println("Testing conventional network before training...")
    acc_conv_before = test_conv(conv_model, test_loader, ps_conv, st_conv) 
    println("Accuracy (Conventional, Before Training): $acc_conv_before")

    println("Training conventional network...")
    losses_conv, ps_conv_trained, st_conv_trained = train(conv_model, ps_conv, st_conv, train_loader, conv_loss_function, args)

    println("Testing conventional network after training...")
    acc_conv_after = test_conv(conv_model, test_loader, ps_conv_trained, st_conv_trained)
    println("Accuracy (Conventional, After Training): $acc_conv_after")

    return acc_conv_after
end

function train_and_test_phasor(ea::ExpArgs)
    # --- Argument/Parameter Setup ---
    println("Setting up parameters...")
    global spk_args
    args = Args(lr = ea.lr, batchsize = 128, epochs = 10, use_cuda = true) # As per notebook

    # Global parameters for spiking tests (from notebook cell 8)
    repeats = 20
    tspan = (0.0, repeats*1.0)

    # Device setup
    device_fn = args.use_cuda ? gpu_device() : cpu_device()
    cdev = cpu_device() # for operations that need to be on CPU like onecold

    # --- Data Loading ---
    println("Loading FashionMNIST dataset...")
    train_data = MLDatasets.FashionMNIST(split=:train)
    test_data = MLDatasets.FashionMNIST(split=:test)

    train_loader = DataLoader(train_data, batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=args.batchsize)

    # --- Phasor Network ---
    global spk_args
    args = Args(lr = ea.lr, batchsize = 128, epochs = 10, use_cuda = true) # As per notebook

    println("\n--- Phasor Network ---")
    act_fn = x -> soft_angle(x, ea.r_lo, ea.r_hi)

    phasor_model = Chain(LayerNorm((28, 28)),
                x -> reshape(x, (28, 28, 1, :)),
                x -> tanh.(x),
                x -> x,  # Placeholder for makespiking in the non-spiking version
                PhasorConv((16, 16), 1 => 3, act_fn),
                PhasorConv((8, 8), 3 => 1, act_fn),
                FlattenLayer(),
                PhasorDense(36 => 128, act_fn),
                Codebook(128 => 10),)

    ps_phasor, st_phasor = Lux.setup(args.rng, phasor_model) .|> device_fn

    # Loss function for phasor network
    function phasor_loss_function(x, y, model, ps, st)
        x = x |> device_fn
        y = y |> device_fn
        y_pred, _ = Lux.apply(model, x, ps, st)
        y_onehot = onehotbatch(y, 0:9)
        loss = codebook_loss(y_pred, y_onehot) 
        loss = mean(loss)
        return loss
    end

    function test_phasor(model, data_loader, ps, st)
        # Evaluation phase
        total_correct = 0
        total_samples = 0
        for (x, y) in data_loader
            x = x |> device_fn
            
            y_pred, _ = Lux.apply(model, x, ps, st)
            pred_labels = predict_codebook(cdev(y_pred))
            
            total_correct += sum(pred_labels .== y .+ 1)
            total_samples += length(y)
        end

        acc = total_correct / total_samples
    end

    println("Testing phasor network before training...")
    # Assuming test_phasor is defined in network_tests.jl
    acc_phasor_before = test_phasor(phasor_model, test_loader, ps_phasor, st_phasor)
    println("Accuracy (Phasor, Before Training): $acc_phasor_before")


    println("Training phasor network...")
    losses_phasor, ps_phasor_trained, st_phasor_trained = train(phasor_model, ps_phasor, st_phasor, train_loader, phasor_loss_function, args, optimiser=RMSProp)

    println("Testing phasor network after training...")
    acc_phasor_after = test_phasor(phasor_model, test_loader, ps_phasor_trained, st_phasor_trained)
    println("Accuracy (Phasor, After Training): $acc_phasor_after")

    # --- Spiking Phasor Network ---
    println("\n--- Spiking Phasor Network ---")
    spk_model = Chain(LayerNorm((28, 28)),
                x -> reshape(x, (28, 28, 1, :)),
                x -> tanh.(x),
                MakeSpiking(spk_args, repeats),
                PhasorConv((16, 16), 1 => 3, act_fn),
                PhasorConv((8, 8), 3 => 1, act_fn),
                FlattenLayer(),
                PhasorDense(36 => 128, act_fn),
                PhasorDense(128 => 10, act_fn))

    

    function fmnist_spiking_accuracy(data_loader, model, ps, st, args)
        acc = []
        n_phases = []
        num = 0

        n_batches = length(data_loader)

        for (x, y) in data_loader
            if args.use_cuda && CUDA.functional()
                x = x |> device_fn
                y = device_fn(1.0f0 .* onehotbatch(y, 0:9))
            end
            
            spk_output, _ = model(x, ps, st)
            ŷ = train_to_phase(spk_output)
            
            append!(acc, sum.(accuracy_quadrature(ŷ, y))) ## Decode the output of the model
            num += size(x)[end]
        end

        acc = sum(reshape(acc, :, n_batches), dims=2) ./ num
        return acc
    end

    # For the spiking model, the notebook uses the trained parameters from the non-spiking phasor network.
    # If you intend to train the spiking model separately, you'd need a different setup and training loop.
    # Here, we'll test it with the trained phasor network parameters as in the notebook.
    # ps_spk, st_spk = Lux.setup(args.rng, spk_model) .|> device_fn # This would be for initial parameters
    if ea.test_spiking
        println("Testing spiking phasor network (using trained phasor network parameters)...")
        # Assuming fmnist_spiking_accuracy is defined in network_tests.jl
        acc_spiking = fmnist_spiking_accuracy(test_loader, spk_model, ps_phasor_trained, st_phasor_trained, args)
        println("Accuracy (Spiking Phasor, using trained non-spiking phasor weights):")
        for (i, acc_val) in enumerate(acc_spiking)
            println("Repeat $(i): $(acc_val)")
        end
        max_spk_acc = maximum(acc_spiking)
        println("Maximum Spiking Accuracy: $max_spk_acc")
        return acc_phasor_after, max_spk_acc
    else
        return acc_phasor_after
    end
    
end

## experimental loop

lrs = [1e-4, 3e-4, 5e-4, 7e-4, 1e-3]
accs_conv = map(lr -> train_and_test_conv(ExpArgs(lr=lr)), lrs)
print(accs_conv)

# accs_phasor = map(lr -> train_and_test_phasor(ExpArgs(lr=lr)), lrs)
# print(accs_phasor)

r_los = [0.1, 0.2, 0.3]
gaps = [0.1, 0.2, 0.3, 0.4]
pairs = collect(Iterators.product(r_los, gaps)) |> vec
pairs = [(r[1], r[1] + r[2]) for r in pairs]

exp_args = map(x -> ExpArgs(lr=x[1], r_lo=x[2][1], r_hi=x[2][2]), Iterators.product(lrs, pairs)) |> vec
accs_phasor = map(ea -> train_and_test_phasor(ea), exp_args)

save(joinpath("runs", "conv_results.jld2"), "accs_conv", accs_conv, 
                        "accs_phasor", accs_phasor,
                        "lrs_conv", lrs,
                        "args_phasor", exp_args)