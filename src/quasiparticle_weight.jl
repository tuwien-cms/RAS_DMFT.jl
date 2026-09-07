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
julia> Σ = PolesSum([1.0], [2.0]);

julia> quasiparticle_weight_inflections(Σ; λmax = 2.0)
1-element Vector{Float64}:
 0.9999999999999998
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

            # bisect for higher accuracy
            for _ in 1:100
                λ_mid = (λ_low + λ_high) / 2
                (λ_mid == λ_low || λ_mid == λ_high) && break
                if sign(_inflection_residual(Σ, tol, λ_mid)) == sign(G_low)
                    λ_low = λ_mid
                else
                    λ_high = λ_mid
                end
            end
            push!(roots, (λ_low + λ_high) / 2)
        end
        G_low, λ_low = G_high, λ_high
    end
    return roots
end

# residual = 0 are the inflection points of Z(λ)
@inline function _inflection_residual(Σ::PolesSum, tol, λ)
    M1, M2, M3 = _regularized_pole_moments(Σ, tol, λ)
    return (1 + M1) * M2 + 4λ^2 * M2^2 - 4λ^2 * (1 + M1) * M3
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
