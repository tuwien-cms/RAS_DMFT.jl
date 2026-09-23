# Various hybridization functions.

# For the Bethe lattice take the  Green's function and rescale by D^/4
# as Δ(ω) = D^2/4 G(ω).

"""
    hybridization_function_bethe_analytic(z::Number, D::Real = 1.0)
    hybridization_function_bethe_analytic(Z::AbstractVector{<:Number}, D::Real = 1.0)

Calculate the hybridization function for a Bethe lattice
given a frequency `z` in the upper complex plane,
and half-bandwidth `D`.

```math
Δ(z) = \\frac{1}{2} \\left(z - \\mathrm{sgn}(\\mathrm{Re}(z)) \\sqrt{z^2 - D^2}\\right)
```

with ``\\mathrm{sgn}(0) = \\mathrm{sgn}(0^±)``.
"""
function hybridization_function_bethe_analytic(z::Number, D::Real = 1.0)
    return greens_function_bethe_analytic(z, D) * D^2 / 4
end

function hybridization_function_bethe_analytic(Z::AbstractVector{<:Number}, D::Real = 1.0)
    return map(z -> hybridization_function_bethe_analytic(z, D), Z)
end

"""
    hybridization_function_bethe_simple(n_bath::Int, D::Real = 1.0)

Return the [`PolesSum`](@ref) representation of the semicircular density of states
with half-bandwidth `D` on `n_bath` poles.

Poles are found by diagonalizing a tridiagonal matrix with hopping ``t=D/2``.

See also
[`greens_function_bethe_grid`](@ref).
"""
function hybridization_function_bethe_simple(n_bath::Int, D::Real = 1.0)
    # Take Green's function and rescale weights by D/2.
    Δ = greens_function_bethe_simple(n_bath, D)
    rmul!(Δ, D^2 / 4)
    return Δ
end

"""
    hybridization_function_bethe_grid(grid::AbstractVector{<:Real}, D::Real = 1.0)

Return the [`PolesSum`](@ref) representation of the semicircular density of states
with half-bandwidth `D` with poles given in `grid`.

See also
[`hybridization_function_bethe_simple`](@ref).
"""
function hybridization_function_bethe_grid(grid::AbstractVector{<:Real}, D::Real = 1.0)
    Δ = greens_function_bethe_grid(grid, D)
    rmul!(Δ, D^2 / 4)
    return Δ
end

"""
    hybridization_function_bethe_grid_hubbard3(
        grid::AbstractVector{<:Real}, U::Real = 0.0, D::Real = 1.0
    )

Return the [`PolesSum`](@ref) representation of the Hubbard III approximation
with half-bandwidth `D` and poles given in `grid`.

Created using two semicircles at ``±U/2``.
"""
function hybridization_function_bethe_grid_hubbard3(
        grid::AbstractVector{<:Real}, U::Real = 0.0, D::Real = 1.0
    )
    Δ = greens_function_bethe_grid_hubbard3(grid, U, D)
    rmul!(Δ, D^2 / 4)
    return Δ
end

# Dispersion relation H_k supplied by user.

"""
    hybridization_function_local(
        H_ks::Vector{<:AbstractMatrix},
        Σ_stat::AbstractMatrix,
        Σ_dyn::PolesSumBlock,
        idx::AbstractVector{<:Integer},
        μ::Real,
        grid::AbstractVector{<:Real};
        tol::Real = 1.0e-8,
    )

Calculate the mean-field level and hybridization function from the lattice.

The correlated orbitals `idx` form the impurity,
with mean-field level ``ϵ_\\mathrm{mf}`` and hybridization function ``Δ``
on the locations `grid`,
obtained without building the pole representation of the local Green's function.

The impurity Green's function is the correlated block of the local one,

```math
G_\\mathrm{imp}(z)
=
P^† \\left\\langle
\\frac{1}{z - M_k - P Σ_\\mathrm{dyn}(z) P^†}
\\right\\rangle P
=
\\frac{1}{z - ϵ_\\mathrm{mf} - Σ_\\mathrm{dyn}(z) - Δ(z)} \\, ,
```

with ``M_k = H_k + Σ_\\mathrm{stat} - μ``
and the k-average ``\\langle A_k \\rangle = \\frac{1}{N_k} ∑_k A_k``.
The mean-field level is the first moment of ``G_\\mathrm{imp}``,

```math
ϵ_\\mathrm{mf} = P^† \\langle M_k \\rangle P \\, .
```

``Δ(z)`` is evaluated in the upper half-plane
through the `n_c × n_c` correction of the correlated block
and projected onto `grid` like [`to_grid`](@ref),
i.e. every pole is split between its two neighbors
conserving the zeroth and the first moment,
and poles outside of `grid` land on the outermost location.
The weights follow from the cumulative distribution ``F_0(x) = ∫_{-∞}^x A_Δ``
and its first moment ``F_1(x) = ∫_{-∞}^x ω A_Δ``,
both computed along the vertical line ``x + \\mathrm{i}y`` as

```math
F_0(x) = \\frac{m_0}{2}
+ \\frac{1}{π} ∫_0^∞ \\mathrm{d}y~\\mathrm{Re}~Δ(x + \\mathrm{i}y) \\, .
```

The total weight of ``Δ`` is ``m_0 = F_0(∞)``,
known exactly from the moments of ``G_\\mathrm{imp}``,

```math
m_0 = P^† \\langle M_k^2 \\rangle P - \\left(P^† \\langle M_k \\rangle P\\right)^2 \\, ,
```

so the weights sum to it by construction.

Returns `(ϵ_mf::Matrix, Δ::PolesSumBlock)`, both `n_c × n_c`.
Take the scalar hybridization with
[`PolesSum(::PolesSumBlock, ::Integer, ::Integer)`](@ref).

# Arguments
- `H_ks::Vector{<:AbstractMatrix}`: Hermitian `n_b × n_b` Hamiltonian per k-point
- `Σ_stat::AbstractMatrix`: static self-energy, `n_b × n_b`
- `Σ_dyn::PolesSumBlock`: dynamic self-energy on the correlated block, `n_c × n_c`
- `idx::AbstractVector{<:Integer}`: the `n_c` correlated orbitals
- `μ::Real`: chemical potential
- `grid::AbstractVector{<:Real}`: sorted locations of the returned poles
- `tol::Real = 1.0e-8`: absolute tolerance of the weight integrals

See also
[`greens_function_local`](@ref).
"""
function hybridization_function_local(
        H_ks::Vector{<:AbstractMatrix},
        Σ_stat::AbstractMatrix,
        Σ_dyn::PolesSumBlock,
        idx::AbstractVector{<:Integer},
        μ::Real,
        grid::AbstractVector{<:Real};
        tol::Real = 1.0e-8,
    )
    # check input
    _, n_c = _check_lattice(H_ks, Σ_stat, Σ_dyn, idx)
    _issorted_and_unique(grid)
    tol > 0 || throw(DomainError(tol, "tol is not positive"))

    n_k = length(H_ks)
    M_ccs, λs, Bs = _schur_bands(H_ks, Σ_stat, idx, μ)
    T = complex(float(promote_type(eltype(first(Bs)), eltype(Σ_dyn))))

    # mean-field level
    ϵ_mf = copy(sum(M_ccs)) # single-element case aliases
    rmul!(ϵ_mf, inv(n_k))
    hermitianpart!(ϵ_mf)
    # zeroth moment
    δMs = [M_cc - ϵ_mf for M_cc in M_ccs] # deviation from the mean
    m0 = sum(δM^2 + B * B' for (δM, B) in zip(δMs, Bs))
    rmul!(m0, inv(n_k))
    hermitianpart!(m0)

    # Δ(z) and z Δ(z) - m0, both O(1/z), free of cancellation at large |z|
    # on the contour to y -> ∞, where z - ϵ_{mf} - Σ_{dyn} - G_{imp}^{-1} loses all digits.
    # With R = ⟨δ_k X_k^{-1}⟩:
    #   Δ = (1 + R)^{-1} R Y,  where R Y = ⟨Δ_k + δ_k X_k^{-1} δ_k⟩ as ⟨δM_k⟩ = 0,
    #   z Δ - m0 = (1 + R)^{-1} (⟨Z_k⟩ - R m0).
    function Δ_moments(z)
        Σ = evaluate(Σ_dyn, z) # shared by all k-points
        Y = z * I - ϵ_mf - Σ
        Rs = Vector{Matrix{T}}(undef, n_k)
        RYs = Vector{Matrix{T}}(undef, n_k)
        Zs = Vector{Matrix{T}}(undef, n_k)
        Threads.@threads for i in 1:n_k
            δM, B, λ = δMs[i], Bs[i], λs[i]
            Δ_k = B * Diagonal(inv.(z .- λ)) * B'
            δ_k = δM + Δ_k
            X_k = Y - δ_k
            R_k = δ_k / X_k
            Rs[i] = R_k
            RYs[i] = Δ_k + R_k * δ_k
            # Z_k = (z Δ_k - B B^†) + (z δ_k X_k^{-1} δ_k - δM^2)
            Zs[i] = B * Diagonal(λ ./ (z .- λ)) * B' + δM * Δ_k + Δ_k * δM + Δ_k^2 +
                R_k * (ϵ_mf + Σ + δ_k) * δ_k
        end
        R = sum(Rs) / n_k
        RY = sum(RYs) / n_k
        Z = sum(Zs) / n_k
        return (I + R) \ RY, (I + R) \ (Z - R * m0) # Δ(z), z Δ(z) - m0
    end

    # F_0(x) and F_1(x) up to their constants, both from one contour integral
    function cumulative(x)
        function integrand(y)
            Δ_z, zΔ_m0 = Δ_moments(x + im * y)
            return hcat(hermitianpart(Δ_z), hermitianpart(zΔ_m0))
        end
        val, _ = quadgk(integrand, 0, Inf; atol = π * tol, maxevals = 10_000)
        val ./= π
        return val[:, 1:n_c] + m0 / 2, val[:, (n_c + 1):end]
    end
    moments = map(cumulative, grid)
    F0s = first.(moments)
    F1s = last.(moments)

    # split every segment between its two grid points, law of levers
    n_g = length(grid)
    wgts = [zeros(T, n_c, n_c) for _ in 1:n_g]
    wgts[1] += F0s[1] # below the grid
    wgts[n_g] += m0 - F0s[n_g] # above the grid
    for s in 1:(n_g - 1)
        h = grid[s + 1] - grid[s]
        ΔF0 = F0s[s + 1] - F0s[s]
        ΔF1 = F1s[s + 1] - F1s[s]
        wgts[s] += (grid[s + 1] * ΔF0 - ΔF1) / h
        wgts[s + 1] += (ΔF1 - grid[s] * ΔF0) / h
    end

    return ϵ_mf, PolesSumBlock(Vector(grid), wgts)
end

# Split `M_k = H_k + Σ_{stat} - μ` into the correlated block `M_{cc,k}`,
# the eigenvalues `λ_k` of the remaining block `M_{rr,k}`,
# and the coupling `B_k = M_{cr,k} U_k` from the correlated block to its eigenvectors,
# so that the downfolded `G_0` is `1 / (z - M_{cc,k} - B_k (z - λ_k)^{-1} B_k^†)`.
function _schur_bands(H_ks, Σ_stat, idx, μ)
    T = float(promote_type(eltype(eltype(H_ks)), eltype(Σ_stat)))
    n_b = LinearAlgebra.checksquare(first(H_ks))
    rest = setdiff(1:n_b, idx)
    n_k = length(H_ks)
    M_ccs = Vector{Matrix{T}}(undef, n_k)
    λs = Vector{Vector{real(T)}}(undef, n_k)
    Bs = Vector{Matrix{T}}(undef, n_k)
    Threads.@threads for k in eachindex(H_ks)
        M = H_ks[k] + Σ_stat - μ * I
        M_ccs[k] = M[idx, idx]
        if isempty(rest)
            # every orbital is correlated, LAPACK rejects the empty block
            λs[k] = real(T)[]
            Bs[k] = Matrix{T}(undef, length(idx), 0)
        else
            F = eigen!(Hermitian(Matrix{T}(M[rest, rest])))
            λs[k] = F.values
            Bs[k] = M[idx, rest] * F.vectors
        end
    end
    return M_ccs, λs, Bs
end
