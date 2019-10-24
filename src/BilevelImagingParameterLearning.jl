module BilevelImagingParameterLearning

using AbstractOperators
using ProximalOperators
using ProximalAlgorithms
using FileIO

# imports
import StructuredOptimization: Term, AbstractExpression
import ProximalOperators: RealOrComplex, ProximableFunction
import AbstractOperators: size, fun_space, fun_name, fun_dom, string_dom, domainType, codomainType
import LinearAlgebra: mul!

# exports
export datasetimage
export mul!

# Norms
#include("norms/NormWL21.jl")

# Operators
include("operators/ScalarROFAdjointOperator.jl")
include("operators/Identity.jl")

# Bindings
include("syntax/terms/bind.jl")

# Printing
function Base.show(io::IO, L::AbstractOperator)
	print(io, fun_name(L)*" "*fun_space(L) )
end

"""
    datasetimage(filename, [ops...])
load test image that partially matches `filename`, the first match is used if there're more than one. If `ops` is specified, it will be passed to `load` function. 

# Example
```julia
julia> using BilevelImagingParameterLearning
julia> datasetimage("cameraman.tif")
julia> datasetimage("cameraman")
julia> datasetimage("c")
```
"""
function datasetimage(filename, imagedir, ops...)
    if !isdir(imagedir)
        @info "Could not find directory for image dataset."
    end

    imagefile = joinpath(imagedir, filename)
    if !isfile(imagefile)
        fls = readdir(imagedir)
        havefile = false
        for f in fls
            if startswith(f, filename)
                imagefile = joinpath(imagedir,f)
                havefile = true
                break
            end
        end

        if !havefile
            @info "Could not find *filename in directory images/."
        end

        img = load(imagefile, ops...)
        Float64.(img) # Forcing the image to be a Float64 array
    end
end
end