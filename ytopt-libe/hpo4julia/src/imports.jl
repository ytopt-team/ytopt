using CUDA, LuxCUDA
using ComponentArrays, Lux
using SciMLSensitivity, DifferentialEquations, Optimisers

using Lux: glorot_uniform, truncated_normal
using LinearAlgebra: diagind, I
using DifferentialEquations: ODESolution
using ChainRulesCore: ignore_derivatives
using Random: GLOBAL_RNG, AbstractRNG
using Interpolations: linear_interpolation
using Statistics: cor, mean
using LinearAlgebra: diag
using OneHotArrays: OneHotMatrix
using NNlib: batched_mul
using Base: @kwdef
using Zygote: withgradient
using Random: Xoshiro

import LuxLib: dropout