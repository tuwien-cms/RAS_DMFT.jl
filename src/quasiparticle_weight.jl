# Quasiparticle weight of the impurity self-energy and its regularization.

"""
    quasiparticle_weight(Σ::PolesSum; tol::Real = 0, λ::Real = 0)

Obtain the quasiparticle weight on the real axis.

```math
\\begin{aligned}
Z &= \\left(1 - \\frac{∂\\mathrm{Re}~Σ(0)}{∂ω}\\right)^{-1} \\\\
  &= \\left(1 - \\sum_i w_i \\frac{∂}{∂ω}\\left.\\frac{1}{ω-a_i}\\right|_{ω=0}\\right)^{-1} \\\\
  &= \\left(1 + \\sum_{w_i ≥ \\mathrm{tol}} w_i\\frac{1}{a_i^2 + λ^2}\\right)^{-1} \\\\
\\end{aligned}
```

Skip all weights ``w_i< \\mathrm{tol}``.
The variable ``λ`` serves as a regularization parameter.
"""
function quasiparticle_weight(Σ::PolesSum; tol::Real = 0, λ::Real = 0)
    tol >= 0 || throw(ArgumentError("tol must be semipositive"))
    λ >= 0 || throw(ArgumentError("λ must be semipositive"))

    deriv = zero(Float64)
    for i in eachindex(Σ)
        weight(Σ, i) < tol && continue
        deriv += weight(Σ, i) / (location(Σ, i)^2 + λ^2)
    end
    return inv(1 + deriv)
end

"""
    quasiparticle_weight_inflections(
        Σ::PolesSum;
        tol::Real = 0,
        λmin::Real = eps(),
        λmax::Real = 1,
    )

Return the inflection points of the regularized quasiparticle weight

```math
Z(λ) = \\left(1 + \\sum_{w_i \\geq \\mathrm{tol}} \\frac{w_i}{a_i^2 + λ^2}\\right)^{-1}
```

in the window ``[λ_\\mathrm{min}, λ_\\mathrm{max}]``.

The inflection points are the solution of ``∂^2Z(λ)/∂λ^2 = 0``,
which are the roots of

```math
(1+M_1) M_2 + 4 λ^2 M_2^2 = 4 λ^2 (1+M_1) M_3,
```

with the regularized pole moments

```math
\\begin{aligned}
M_1 &= \\sum_{w_i \\geq \\mathrm{tol}} \\frac{w_i}{a_i^2 + λ^2}, \\\\
M_2 &= \\sum_{w_i \\geq \\mathrm{tol}} \\frac{w_i}{(a_i^2 + λ^2)^2}, \\\\
M_3 &= \\sum_{w_i \\geq \\mathrm{tol}} \\frac{w_i}{(a_i^2 + λ^2)^3}.
\\end{aligned}
```

# Examples
```jldoctest
julia> Σ = PolesSum([1.0e-4, 1.0], [1.0e-7, 2.0]);

julia> quasiparticle_weight_inflections(Σ; λmax = 2.0)
3-element Vector{Float64}:
 0.00012018504652954801
 0.019681873959735308
 0.9999998333332845
```

See also [`quasiparticle_weight`](@ref).
"""
function quasiparticle_weight_inflections(
        Σ::PolesSum;
        tol::Real = 0,
        λmin::Real = eps(),
        λmax::Real = 1,
    )

    # check input
    tol >= 0 || throw(ArgumentError("tol must be semipositive"))
    λmax > 0 || throw(ArgumentError("λmax must be positive"))
    λmin > 0 || throw(ArgumentError("λmin must be positive"))
    λmin < λmax || throw(ArgumentError("violating λmin < λmax"))

    TΣ = float(eltype(Σ))

    λs = logrange(λmin, λmax; length = 10_000) # enough points per decade

    roots = TΣ[]
    λ_low = λs[1]
    G_low = _inflection_residual(Σ, tol, λ_low)
    @inbounds for λ_high in λs[2:end]
        G_high = _inflection_residual(Σ, tol, λ_high)
        if sign(G_low) != sign(G_high)
            push!(
                roots,
                _bisect_sign_change(λ -> _inflection_residual(Σ, tol, λ), λ_low, λ_high)
            )
        end
        G_low, λ_low = G_high, λ_high
    end
    return roots
end

"""
    quasiparticle_weight_optimum_regularization(
        Σ::PolesSum;
        tol::Real = 0,
        λmin::Real = eps(),
        λmax::Real = 1,
        prominence::Real = 10,
    )

Return the regularization parameter ``λ`` of the first plateau of
[`quasiparticle_weight`](@ref) in ``[λ_\\mathrm{min}, λ_\\mathrm{max}]``.

A plateau is a minimum of the slope on a logarithmic ``λ`` axis: ``∂^2Z/∂(\\ln λ)^2 = 0``.
It only counts if the rise separating it from the plateau at ``λ → 0`` is
at least `prominence` times steeper than the plateau itself, which discards wiggles.
Return zero if there is no such plateau, as ``Z(0)`` is then optimal.

# Examples
```jldoctest
julia> Σ = PolesSum([1.0e-4, 1.0], [1.0e-7, 2.0]); # tiny pole close to zero

julia> quasiparticle_weight_optimum_regularization(Σ)
0.014951703474155864

julia> quasiparticle_weight_optimum_regularization(PolesSum([1.0], [2.0]))
0.0
```

See also [`quasiparticle_weight`](@ref).
"""
function quasiparticle_weight_optimum_regularization(
        Σ::PolesSum;
        tol::Real = 0,
        λmin::Real = eps(),
        λmax::Real = 1,
        prominence::Real = 10,
    )

    # check input
    tol >= 0 || throw(ArgumentError("tol must be semipositive"))
    λmax > 0 || throw(ArgumentError("λmax must be positive"))
    λmin > 0 || throw(ArgumentError("λmin must be positive"))
    λmin < λmax || throw(ArgumentError("violating λmin < λmax"))
    prominence >= 1 || throw(ArgumentError("prominence must be at least one"))

    TΣ = float(eltype(Σ))
    λs = logrange(λmin, λmax; length = 10_000) # enough points per decade

    # steepest rise so far, the reference for the prominence of a plateau
    slope_max = _quasiparticle_weight_log_slope(Σ, tol, λmin)
    λ_low = λmin
    residual_low = _quasiparticle_weight_log_curvature(Σ, tol, λ_low)
    @inbounds for λ_high in λs[2:end]
        residual_high = _quasiparticle_weight_log_curvature(Σ, tol, λ_high)
        # a minimum of the logarithmic slope turns the residual from - to +
        if residual_low < 0 <= residual_high
            λ_plateau = _bisect_sign_change(
                λ -> _quasiparticle_weight_log_curvature(Σ, tol, λ), λ_low, λ_high
            )
            slope_plateau = _quasiparticle_weight_log_slope(Σ, tol, λ_plateau)
            slope_max / slope_plateau >= prominence && return λ_plateau
        end
        slope_max = max(slope_max, _quasiparticle_weight_log_slope(Σ, tol, λ_high))
        residual_low, λ_low = residual_high, λ_high
    end
    return zero(TΣ)
end

# ∂Z(λ)/∂λ
function _quasiparticle_weight_slope(Σ::PolesSum, tol, λ)
    M1, M2, _ = _regularized_pole_moments(Σ, tol, λ)
    Z = inv(1 + M1)
    return 2λ * Z^2 * M2
end

# residual = 0 are the inflection points of Z(λ)
@inline function _inflection_residual(Σ::PolesSum, tol, λ)
    M1, M2, M3 = _regularized_pole_moments(Σ, tol, λ)
    return (1 + M1) * M2 + 4λ^2 * M2^2 - 4λ^2 * (1 + M1) * M3
end

# slope (∂ Z)/(∂ logλ)
function _quasiparticle_weight_log_slope(Σ::PolesSum, tol, λ)
    M1, M2, _ = _regularized_pole_moments(Σ, tol, λ)
    Z = inv(1 + M1)
    return 2 * λ^2 * Z^2 * M2
end

# proportional to curvature (∂^2 Z)/(∂(logλ)^2)
# common factor 4 λ^2 Z^2 canceled as only the sign is relevant
@inline function _quasiparticle_weight_log_curvature(Σ::PolesSum, tol, λ)
    M1, M2, M3 = _regularized_pole_moments(Σ, tol, λ)
    return (1 + M1) * M2 + 2 * λ^2 * M2^2 - 2 * λ^2 * (1 + M1) * M3
end

# bisect the sign change of `residual` bracketed by [λ_low, λ_high]
function _bisect_sign_change(residual, λ_low, λ_high)
    sign_low = sign(residual(λ_low))
    for _ in 1:60  # enough for Float64 machine precision (53 bits)
        λ_mid = (λ_low + λ_high) * 0.5
        (λ_mid == λ_low || λ_mid == λ_high) && break
        if sign(residual(λ_mid)) == sign_low
            λ_low = λ_mid
        else
            λ_high = λ_mid
        end
    end
    return (λ_low + λ_high) * 0.5
end

# regularized pole moments skipping weights below tol
@inline function _regularized_pole_moments(Σ::PolesSum, tol, λ)
    M1 = M2 = M3 = zero(float(eltype(Σ)))
    for (loc, wgt) in Σ
        wgt < tol && continue
        invden = inv(loc^2 + λ^2)
        M1 += wgt * invden
        M2 += wgt * invden^2
        M3 += wgt * invden^3
    end
    return M1, M2, M3
end
