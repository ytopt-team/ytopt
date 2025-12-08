#=
scripts/train_fashionmnist.jl

Run from repository root like:

julia scripts/train_fashionmnist.jl --lr 0.001 --epochs 5 --optimizer rmsprop --batchsize 128 --use_cuda true

This script:
 1. Activates the project environment
 2. Loads FashionMNIST using MLDatasets and creates DataLoaders
 3. Trains and evaluates a conventional Lux model and the PhasorNetworks model

Notes:
 - It reuses the project's PhasorNetworks code via `include("../src/PhasorNetworks.jl")`.
 - CLI parsing uses Base.ArgParse-like simple manual parsing to avoid extra dependencies.
=#

using Pkg
# activate repository project automatically detecting the repository root
function find_repo_root(start_dir::String = pwd())
    dir = start_dir
    # Loop until we find a directory that contains both Project.toml and a .git folder
    while !(isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, ".git")))
        parent = dirname(dir)
        if parent == dir
            error("Repository root not found from $(start_dir)")
        end
        dir = parent
    end
    return dir
end

repo_root = find_repo_root(@__DIR__)
cd(repo_root)  # change working directory to repository base
Pkg.activate(repo_root)

# bring project code into scope
include("../src/PhasorNetworks.jl")
using .PhasorNetworks

# dependencies used in notebook
using Lux, MLUtils, MLDatasets, OneHotArrays, Statistics, Random, Zygote, Optimisers, ComponentArrays, CUDA
using ArgParse, JLD2, Dates
using Random: Xoshiro

# parse CLI args with ArgParse
function build_parser()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--lr"
            help = "Learning rate"
            arg_type = Float64
            default = 0.001
        "--epochs"
            help = "Number of training epochs"
            arg_type = Int
            default = 5
        "--batchsize"
            help = "Batch size"
            arg_type = Int
            default = 128
        "--seed"
            help = "RNG seed"
            arg_type = Int
            default = 42
        "--optimizer"
            help = "Optimizer name (rmsprop, adam, sgd)"
            default = "rmsprop"
        "--use_cuda"
            help = "Use CUDA if available"
            arg_type = Bool
            default = true
        "--out"
            help = "Output directory to save run artifacts"
            default = "runs"
    end
    return s
end

parsed = parse_args(build_parser())
lr = parsed["lr"]
epochs = parsed["epochs"]
seed = parsed["seed"]
batchsize = parsed["batchsize"]
optimizer_name = lowercase(parsed["optimizer"])
use_cuda = parsed["use_cuda"]
outdir = parsed["out"]


args = Args(batchsize = batchsize,
            epochs = epochs,
            lr = lr,
            rng = Xoshiro(seed),
            use_cuda = use_cuda)

println("Settings: lr=$(lr), epochs=$(epochs), batchsize=$(batchsize), optimizer=$(optimizer_name), use_cuda=$(use_cuda), out=$(outdir)")


# devices
cdev = cpu_device()
gdev = gpu_device()

if use_cuda
    dev = gdev
else 
    dev = cdev
end

function get_optimizer(name::String, lr::Float64)
    name = lowercase(name)
    if name in ("rmsprop", "rmsprop()")
        return Optimisers.RMSProp
    elseif name in ("adam", "adam()")
        return Optimisers.Adam
    elseif name in ("sgd", "sgd()")
        return Optimisers.SGD
    else
        error("Unknown optimizer: $name. Supported: rmsprop, adam, sgd")
    end
end

optimiser = get_optimizer(optimizer_name, lr)

# load data
println("Loading FashionMNIST...")
train_data = MLDatasets.FashionMNIST(split=:train)
test_data = MLDatasets.FashionMNIST(split=:test)
train_loader = DataLoader(train_data, batchsize=batchsize)
test_loader = DataLoader(test_data, batchsize=batchsize)

# conventional model
construct_model = n -> Chain(FlattenLayer(),
                        LayerNorm((28^2,)),
                        Dense(28^2 => n, relu),
                        Dense(n => 10),
                        softmax)

function loss_function(x, y, model, ps, st, dev=dev)
    x = x |> dev
    y = y |> dev
    y_pred, _ = Lux.apply(model, x, ps, st)
    y_onehot = onehotbatch(y, 0:9)
    return CrossEntropyLoss(;logits=false, dims=1)(y_pred, y_onehot)
end

function test(model, data_loader, ps, st, dev=dev)
    total_correct = 0
    total_samples = 0
    for (x, y) in data_loader
        x = x |> dev
        y_pred, _ = Lux.apply(model, x, ps, st)
        pred_labels = onecold(cdev(y_pred))
        total_correct += sum(pred_labels .== y .+ 1)
        total_samples += length(y)
    end
    return total_correct / total_samples
end

# training utility using the project's train function if present, otherwise simple loop
function run_training(model, ps, st, train_loader, loss_fn, args)
    if isdefined(Main, :train)
        return train(model, ps, st, train_loader, loss_fn, args; optimiser=optimiser)
    else
        error("No train function found in environment. Please use the project's train implementation.")
    end
end

# Conventional model run
println("\n=== Conventional network ===")
model = construct_model(128)
ps, st = Lux.setup(args.rng, model)
ps = ps |> dev
st = st |> dev

println("Initial loss (first batch): ", loss_function(first(train_loader)..., model, ps, st))

losses, pst, stt = run_training(model, ps, st, train_loader, loss_function, args)
println("Conventional final loss: ", losses[end])
acc = test(model, test_loader, pst, stt)
println("Conventional test accuracy: ", acc)

# Phasor model
println("\n=== Phasor network ===")
import .PhasorNetworks: default_bias, Codebook
p_model = Chain(FlattenLayer(),
                LayerNorm((28^2,)),
                x -> tanh.(x),
                x -> x,
                PhasorDense(28^2 => 128, soft_angle, init_bias=default_bias),
                PhasorDense(128 => 16, soft_angle, init_bias=default_bias),
                Codebook(16 => 10))

psp, stp = Lux.setup(args.rng, p_model)

if use_cuda && CUDA.functional()
    psp = psp |> gdev
    stp = stp |> gdev
end

function phasor_loss_function(x, y, model, ps, st, dev=dev)
    x = x |> dev
    y = y |> dev
    y_pred, _ = Lux.apply(model, x, ps, st)
    y_onehot = onehotbatch(y, 0:9)
    loss = evaluate_loss(y_pred, y_onehot, :similarity) 
    loss = mean(loss)
    return loss
end
function test_phasor(model, data_loader, ps, st, dev=dev)
    # Evaluation phase
    total_correct = 0
    total_samples = 0
    for (x, y) in data_loader
        x = x |> dev
        
        y_pred, _ = Lux.apply(model, x, ps, st)
        pred_labels = predict(cdev(y_pred), :similarity)
        
        total_correct += sum(pred_labels .== y .+ 1)
        total_samples += length(y)
    end

    acc = total_correct / total_samples
end

println("Initial phasor loss (first batch): ", phasor_loss_function(first(train_loader)..., p_model, psp, stp))

losses_f, ps_train_f, st_train_f = run_training(p_model, psp, stp, train_loader, phasor_loss_function, args)
println("Phasor final loss: ", losses_f[end])
acc_p = test_phasor(p_model, test_loader, ps_train_f, st_train_f)
println("Phasor test accuracy (codebook prediction): ", acc_p)

println("\nDone.")


# --- Save run artifacts (loss histories, trained params, args) using JLD2

function cpuify(x)
    # move CUDA arrays to CPU; recursively handle arrays, tuples, dicts, named tuples
    if typeof(x) <: CUDA.CuArray
        return Array(x)
    elseif isa(x, AbstractArray)
        return map(cpuify, x)
    elseif isa(x, Dict)
        out = Dict()
        for (k,v) in x
            out[k] = cpuify(v)
        end
        return out
    elseif isa(x, NamedTuple)
        return NamedTuple{keys(x)}(cpuify.(values(x)))
    else
        return x
    end
end

function save_run(outdir, info)
    # create output dir
    if !isdir(outdir)
        mkpath(outdir)
    end
    ts = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    fname = joinpath(outdir, "fmnist_run_$(ts).jld2")
    @info "Saving run to $fname"
    @save fname info
    return fname
end

# gather info
info = Dict(
    :args => Dict(:lr=>lr, :epochs=>epochs, :batchsize=>batchsize, :optimizer=>optimizer_name, :use_cuda=>use_cuda, :seed=>seed),
    :conventional => Dict(:losses => cpuify(losses), :params => cpuify(pst), :state => cpuify(stt), :test_accuracy => acc),
    :phasor => Dict(:losses => cpuify(losses_f), :params => cpuify(ps_train_f), :state => cpuify(st_train_f), :test_accuracy => acc_p)
)

save_path = save_run(outdir, info)
println("Saved run artifacts to: ", save_path)
