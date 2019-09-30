# BilevelImagingParameterLearning.jl
Operators and Norms involved in the formulation of variational imaging problems. It is an extension of the StructuredOptimization julia module, designed to support bilevel programming problems

## Installation
To install the package, hit `]` from the Julia command line to enter the package manager, then

```julia
pkg> add BilevelImagingParameterLearning
```

## Description
These are a set of norms used for imaging models, in particular:
1.  Scalar Total Variation (TV)
2.  Scale-Dependent Total Variation (SD-TV)
3.  Piecewise-Constant Total Variation (PC-TV)
4.  Generalized Total Variation (TGV)

These are a set of operators used for imaging models, in particular:
1.  SecondOrderVariation
2.  FFT

## Usage
With `using BilevelImagingParameterLearning` the package exports the `eval` and `eval!` methods to evaluate an argument, `conj` and `conj!` methods to evaluate the conjugated mapping of several inputs.

For example, you can create the L1-norm as follows.

```julia
julia> f = NormL1(3.5)
description : weighted L1 norm
type        : Array{Complex} → Real
expression  : x ↦ λ||x||_1
parameters  : λ = 3.5
```

Functions created this way are, of course, callable.

```julia
julia> x = randn(10) # some random point
julia> f(x)
32.40700818735099
```

## References
1. De Los Reyes, Villacis, "Bilevel Scale Dependent ROF Parameter Learning", XXX