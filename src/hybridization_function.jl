# Various hybridization functions.

# For the Bethe lattice take the Green's function and rescale by D^2/4
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

with ``\\mathrm{sgn}(0) = \\mathrm{sgn}(0^+)``.
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
[`hybridization_function_bethe_grid`](@ref).
"""
function hybridization_function_bethe_simple(n_bath::Int, D::Real = 1.0)
    # Take Green's function and rescale weights by D^2/4.
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
        Σ_dyns::AbstractVector{<:PolesSumBlock},
        idxs::AbstractVector{<:AbstractVector{<:Integer}},
        μ::Real,
        grid::AbstractVector{<:Real};
        tol::Real = 1.0e-8,
    )

Calculate the mean-field level and hybridization function from the lattice
for every site `s`, i.e. the group of correlated orbitals `idxs[s]`.

Every site ``s`` is an impurity
with dynamic self-energy ``Σ_{\\mathrm{dyn},s}`` = `Σ_dyns[s]`,
mean-field level ``ϵ_{\\mathrm{mf},s}``, and hybridization function ``Δ_s``
on the locations `grid`,
obtained without building the pole representation of the local Green's function.
The `n_c` correlated orbitals of all sites carry the dynamic self-energy
``Σ_\\mathrm{dyn}``, block-diagonal with the blocks ``Σ_{\\mathrm{dyn},s}``.

The impurity Green's function of site ``s`` is its block of the local one,

```math
G_{\\mathrm{imp},s}(z)
=
\\left[
P^† \\left\\langle
\\frac{1}{z - M_k - P Σ_\\mathrm{dyn}(z) P^†}
\\right\\rangle P
\\right]_{ss}
=
\\frac{1}{z - ϵ_{\\mathrm{mf},s} - Σ_{\\mathrm{dyn},s}(z) - Δ_s(z)} \\, ,
```

with ``M_k = H_k + Σ_\\mathrm{stat} - μ``
and the k-average ``\\langle A_k \\rangle = \\frac{1}{N_k} ∑_k A_k``,
so the self-energies of the other sites enter ``Δ_s``.
The mean-field level is the first moment of ``G_{\\mathrm{imp},s}``,

```math
ϵ_{\\mathrm{mf},s} = \\left(P^† \\langle M_k \\rangle P\\right)_{ss} \\, .
```

``Δ_s(z)`` is evaluated in the upper half-plane
through the `n_c × n_c` correction of the correlated block
and projected onto `grid` like [`to_grid`](@ref),
i.e. every pole is split between its two neighbors
conserving the zeroth and the first moment,
and poles outside of `grid` land on the outermost location.
The weights follow from the cumulative distribution ``F_0(x) = ∫_{-∞}^x A_{Δ_s}``
and its first moment ``F_1(x) = ∫_{-∞}^x ω A_{Δ_s}``,
both computed along the vertical line ``x + \\mathrm{i}y`` as

```math
F_0(x) = \\frac{m_{0,s}}{2}
+ \\frac{1}{π} ∫_0^∞ \\mathrm{d}y~\\mathrm{Re}~Δ_s(x + \\mathrm{i}y) \\, .
```

The total weight of ``Δ_s`` is ``m_{0,s} = F_0(∞)``,
known exactly from the moments of ``G_{\\mathrm{imp},s}``,

```math
m_{0,s} = \\left(P^† \\langle M_k^2 \\rangle P\\right)_{ss} - ϵ_{\\mathrm{mf},s}^2 \\, ,
```

so the weights sum to it by construction.
It exceeds the site block of the cluster weight,
the single-site method on all of `idxs`,
by the averaged inter-site hopping squared
``∑_{t ≠ s} \\langle M_k \\rangle_{st} \\langle M_k \\rangle_{st}^†``,
as the other sites belong to the bath of site ``s``.

Returns `(ϵ_mfs::Vector{<:Matrix}, Δs::Vector{<:PolesSumBlock})`
with the entries ``ϵ_{\\mathrm{mf},s}`` and ``Δ_s``,
both `n_s × n_s` for the `n_s` orbitals in `idxs[s]`.
Take a scalar hybridization with
[`PolesSum(::PolesSumBlock, ::Integer, ::Integer)`](@ref).

# Arguments
- `H_ks::Vector{<:AbstractMatrix}`: Hermitian `n_b × n_b` Hamiltonian per k-point
- `Σ_stat::AbstractMatrix`: static self-energy, `n_b × n_b`
- `Σ_dyns::AbstractVector{<:PolesSumBlock}`: dynamic self-energy per site, `n_s × n_s`
- `idxs::AbstractVector{<:AbstractVector{<:Integer}}`: the `n_s` correlated orbitals
  per site, no orbital in two sites
- `μ::Real`: chemical potential
- `grid::AbstractVector{<:Real}`: sorted locations of the returned poles
- `tol::Real = 1.0e-8`: absolute tolerance of the weight integrals

See also
[`greens_function_local`](@ref).
"""
function hybridization_function_local(
        H_ks::Vector{<:AbstractMatrix},
        Σ_stat::AbstractMatrix,
        Σ_dyns::AbstractVector{<:PolesSumBlock},
        idxs::AbstractVector{<:AbstractVector{<:Integer}},
        μ::Real,
        grid::AbstractVector{<:Real};
        tol::Real = 1.0e-8,
    )
    # check input
    length(Σ_dyns) == length(idxs) ||
        throw(DimensionMismatch("Σ_dyns and idxs have different lengths"))
    for (Σ_dyn, idx) in zip(Σ_dyns, idxs)
        _check_lattice(H_ks, Σ_stat, Σ_dyn, idx)
    end
    idx = reduce(vcat, idxs)
    allunique(idx) || throw(ArgumentError("idxs share orbitals"))
    _issorted_and_unique(grid)
    tol > 0 || throw(DomainError(tol, "tol is not positive"))

    n_c = length(idx)
    sites = [findall(in(i), idx) for i in idxs] # positions of every site in idx
    n_k = length(H_ks)
    M_ccs, λs, Bs = _schur_bands(H_ks, Σ_stat, idx, μ)
    T_Σ = mapreduce(eltype, promote_type, Σ_dyns)
    T = complex(float(promote_type(eltype(first(Bs)), T_Σ)))

    # mean-field level, block-diagonal in sites
    M_avg = copy(sum(M_ccs)) # single-element case aliases
    rmul!(M_avg, inv(n_k))
    ϵ_mf = _blockdiag(hermitianpart!(M_avg), sites)
    # zeroth moment, its diagonal blocks are those of every site
    δMs = [M_cc - ϵ_mf for M_cc in M_ccs] # deviation from the mean, ⟨δM⟩_ss = 0
    m0_full = sum(δM^2 + B * B' for (δM, B) in zip(δMs, Bs))
    rmul!(m0_full, inv(n_k))
    m0 = _blockdiag(hermitianpart!(m0_full), sites)

    # Δ(z) and z Δ(z) - m0, both O(1/z), free of cancellation at large |z|
    # on the contour to y -> ∞, where z - ϵ_{mf} - Σ_{dyn} - G_{imp}^{-1} loses all digits.
    # With R = ⟨δ_k X_k^{-1}⟩ and Y block-diagonal, per site s:
    #   Δ = (1 + R)_ss^{-1} (R Y)_ss,
    #     where (R Y)_ss = ⟨Δ_k + δ_k X_k^{-1} δ_k⟩_ss as ⟨δM_k⟩_ss = 0,
    #   z Δ - m0 = (1 + R)_ss^{-1} (⟨Z_k⟩ - R m0)_ss,  as m0 is block-diagonal.
    function Δ_moments(z)
        Σ = zeros(T, n_c, n_c) # shared by all k-points
        for (s, Σ_dyn) in zip(sites, Σ_dyns)
            Σ[s, s] = evaluate(Σ_dyn, z)
        end
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
        A = I + R
        ZRm0 = Z - R * m0
        Δ_z = zeros(T, n_c, n_c)
        zΔ_m0 = zeros(T, n_c, n_c)
        for s in sites
            Δ_z[s, s] = A[s, s] \ RY[s, s]
            zΔ_m0[s, s] = A[s, s] \ ZRm0[s, s]
        end
        return Δ_z, zΔ_m0 # Δ(z), z Δ(z) - m0
    end

    # F_0(x) and F_1(x) up to their constants, both from one contour integral
    function cumulative(x)
        function integrand(y)
            Δ_z, zΔ_m0 = Δ_moments(x + im * y)
            return hcat(hermitianpart(Δ_z), hermitianpart(zΔ_m0))
        end
        atol = π * tol
        val, err = quadgk(integrand, 0, Inf; atol, maxevals = 10_000)
        err > atol && @warn "hybridization integral missed tol" x err = err / π tol
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

    ϵ_mfs = [ϵ_mf[s, s] for s in sites]
    Δs = [PolesSumBlock(Vector(grid), [w[s, s] for w in wgts]) for s in sites]
    return ϵ_mfs, Δs
end

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

Calculate the mean-field level and hybridization function of the single site `idx`
with dynamic self-energy `Σ_dyn`.

This is the [method for several sites](@ref hybridization_function_local(::Vector{<:AbstractMatrix}, ::AbstractMatrix, ::AbstractVector{<:PolesSumBlock}, ::AbstractVector{<:AbstractVector{<:Integer}}, ::Real, ::AbstractVector{<:Real}))
with `[Σ_dyn]` and `[idx]`,
returning `(ϵ_mf::Matrix, Δ::PolesSumBlock)`, both `n_c × n_c`,
instead of vectors of one entry.
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
    ϵ_mfs, Δs = hybridization_function_local(H_ks, Σ_stat, [Σ_dyn], [idx], μ, grid; tol)
    return only(ϵ_mfs), only(Δs)
end

# copy of `A` with the entries between different blocks of `sites` set to zero
function _blockdiag(A, sites)
    B = zeros(eltype(A), size(A))
    for s in sites
        B[s, s] = A[s, s]
    end
    return B
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
