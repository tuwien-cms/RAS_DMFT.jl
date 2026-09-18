"""
    AbstractPoles{A, B}

Supertype which represents a function on the real axis as a collection of poles.

The locations are described by `A`, the weights/amplitudes by `B`.
"""
abstract type AbstractPoles{A, B} end

"""
    amplitude(P::AbstractPoles, i::Integer)

Return the amplitude `B_i` of `P` at index `i`,
satisfying ``B_i B_i^† = W_i`` for the weight `W_i`.

See also [`amplitudes`](@ref).
"""
function amplitude end

"""
    amplitudes(P::AbstractPoles)

Return the amplitude at every index of `P`.

See also [`amplitude`](@ref).
"""
function amplitudes end

"""
    evaluate(P::AbstractPoles, z)

Evaluate `P` at the complex variable `z`.

See also [`evaluate_gaussian`](@ref), [`evaluate_lorentzian`](@ref).
"""
function evaluate end
function evaluate(P::AbstractPoles, z::AbstractVector{<:Number})
    return map(i -> evaluate(P, i), z)
end

"""
    evaluate_gaussian(P::AbstractPoles, ω, σ)

Evaluate `P` with Gaussian broadening ``σ``.
"""
function evaluate_gaussian end
function evaluate_gaussian(P::AbstractPoles, ω::AbstractVector{<:Real}, σ)
    return map(i -> evaluate_gaussian(P, i, σ), ω)
end

"""
    evaluate_lorentzian(P::AbstractPoles, ω, δ)

Evaluate `P` with Lorentzian broadening ``P(ω + \\mathrm{i}δ)``.
"""
evaluate_lorentzian(P::AbstractPoles, ω::Real, δ::Real) = evaluate(P, ω + im * δ)
function evaluate_lorentzian(P::AbstractPoles, ω::AbstractVector{<:Real}, δ)
    return map(i -> evaluate_lorentzian(P, i, δ), ω)
end

"""
    location(P::AbstractPoles, i::Integer)

Return the location of `P` at index `i`.

See also [`locations`](@ref).
"""
location(P::AbstractPoles, i::Integer) = locations(P)[i]

"""
    locations(P::AbstractPoles)

Return the locations of `P`.

See also [`location`](@ref).
"""
locations(P::AbstractPoles) = P.locations

"""
    weight(P::AbstractPoles, i::Integer)

Return the weight of `P` at index `i`.

See also [`weights`](@ref).
"""
function weight end

"""
    weights(P::AbstractPoles)

Return the weights of `P`.

See also [`weight`](@ref).
"""
function weights end

# Gaussian broadening of pole
function _gaussian_broadened(ω, loc, σ)
    re = sqrt(2) / σ * dawson((ω - loc) / (sqrt(2) * σ))
    im_part = pdf(Normal(loc, σ), ω)
    return re - im * π * im_part
end

function _show_poles(io::IO, P::AbstractPoles)
    print(io, summary(P), " with ", length(P), " poles")
    return nothing
end

Base.eltype(::Type{<:AbstractPoles{A, B}}) where {A, B} = promote_type(A, B)
Base.isempty(P::AbstractPoles) = iszero(length(P))
Base.length(P::AbstractPoles) = length(locations(P))
Base.show(io::IO, P::AbstractPoles) = _show_poles(io, P)
