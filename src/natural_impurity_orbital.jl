"""
    NaturalImpurityOrbital{T<:Real}

Natural impurity orbital representation of a single-particle Hamiltonian.

The ordering is `[i, b, v_1, ..., v_(n_v), c_1, ..., c_(n_c)]`,
where `i` is the impurity,
`b` its mirror,
`v_1...v_(n_v)` the fully occupied valence chain,
and `c_1...c_(n_c)` the empty conduction chain.

# Fields
- `H_ib`: 2×2 impurity-mirror Hamiltonian block
- `i_v`: Impurity ↔ valence coupling (i↔v₁)
- `i_c`: Impurity ↔ conduction coupling (i↔c₁)
- `b_v`: Mirror ↔ valence coupling (b↔v₁)
- `b_c`: Mirror ↔ conduction coupling (b↔c₁)
- `e_v`: Valence on-site energies
- `t_v`: Valence hoppings (v₁↔v₂↔...)
- `e_c`: Conduction on-site energies
- `t_c`: Conduction hoppings (c₁↔c₂↔...)

# See also
`natural_impurity_orbital` to construct.
"""
struct NaturalImpurityOrbital{T <: Real}
    H_ib::SMatrix{2, 2, T, 4}
    i_v::T
    i_c::T
    b_v::T
    b_c::T
    e_v::Vector{T}
    t_v::Vector{T}
    e_c::Vector{T}
    t_c::Vector{T}

    function NaturalImpurityOrbital{T}(
            H_ib,
            i_v,
            i_c,
            b_v,
            b_c,
            e_v,
            t_v,
            e_c,
            t_c
        ) where {T}
        issymmetric(H_ib) || throw(ArgumentError("H_ib is not symmetric"))
        all(<(0), e_v)::Bool || throw(ArgumentError("positive valence energies"))
        length(t_v) == max(length(e_v) - 1, 0) ||
            throw(ArgumentError("length mismatch in valence sites and hopping"))
        all(>(0), e_c)::Bool || throw(ArgumentError("negative conduction energies"))
        length(t_c) == max(length(e_c) - 1, 0) ||
            throw(ArgumentError("length mismatch in conduction sites and hopping"))
        return new{T}(H_ib, i_v, i_c, b_v, b_c, e_v, t_v, e_c, t_c)
    end
end

function NaturalImpurityOrbital(
        H_ib::SMatrix{2, 2, T},
        i_v::T,
        i_c::T,
        b_v::T,
        b_c::T,
        e_v::Vector{T},
        t_v::Vector{T},
        e_c::Vector{T},
        t_c::Vector{T},
    ) where {T <: Real}
    return NaturalImpurityOrbital{T}(H_ib, i_v, i_c, b_v, b_c, e_v, t_v, e_c, t_c)
end

n_conduction(H_nat::NaturalImpurityOrbital) = length(H_nat.e_c)

n_valence(H_nat::NaturalImpurityOrbital) = length(H_nat.e_v)

Base.eltype(::Type{NaturalImpurityOrbital{T}}) where {T} = T

# Reconstruct the full n×n Hamiltonian matrix (for tests, debugging).
function Base.Matrix(H_nat::NaturalImpurityOrbital{T}) where {T}
    n_v = n_valence(H_nat)
    n_c = n_conduction(H_nat)
    M = zeros(T, size(H_nat))

    # Impurity-mirror block
    M[1:2, 1:2] = H_nat.H_ib

    # Valence chain couplings (i,b ↔ v₁)
    if n_v >= 1
        M[1, 3] = M[3, 1] = H_nat.i_v
        M[2, 3] = M[3, 2] = H_nat.b_v
    end

    # Conduction chain couplings (i,b ↔ c₁)
    if n_c >= 1
        j = 3 + n_v
        M[1, j] = M[j, 1] = H_nat.i_c
        M[2, j] = M[j, 2] = H_nat.b_c
    end

    # Valence chain (tridiagonal)
    for i in 1:n_v
        M[2 + i, 2 + i] = H_nat.e_v[i]
    end
    for i in 1:(n_v - 1)
        M[2 + i, 3 + i] = M[3 + i, 2 + i] = H_nat.t_v[i]
    end

    # Conduction chain (tridiagonal)
    for i in 1:n_c
        M[2 + n_v + i, 2 + n_v + i] = H_nat.e_c[i]
    end
    for i in 1:(n_c - 1)
        M[2 + n_v + i, 3 + n_v + i] =
            M[3 + n_v + i, 2 + n_v + i] = H_nat.t_c[i]
    end

    return M
end

Base.show(io::IO, H_nat::NaturalImpurityOrbital) =
    print(io, join(size(H_nat), '×'), " ", typeof(H_nat))
function Base.show(io::IO, ::MIME"text/plain", H_nat::NaturalImpurityOrbital)
    println(io, join(size(H_nat), '×'), " ", typeof(H_nat), ":")
    Base.print_matrix(io, Matrix(H_nat))
    return nothing
end

function Base.size(H_nat::NaturalImpurityOrbital)
    n = 2 + n_valence(H_nat) + n_conduction(H_nat)
    return (n, n)
end

Base.size(H_nat::NaturalImpurityOrbital, d::Integer) = d <= 2 ? size(H_nat)[d] : 1

"""
    natural_impurity_orbital(Δ::PolesSum, ϵ_mf::Real; tol::Real = 1.0e-8)

Transform the hybridization function `Δ` and mean-field impurity energy `ϵ_mf` to
the natural impurity orbital basis.

The basis is built from the mean-field reference ``G(z) = 1/(z - ϵ_mf - Δ(z))``.
Weight within `tol` of zero is shared evenly between valence and conduction sites.
"""
function natural_impurity_orbital(Δ::PolesSum, ϵ_mf::Real; tol::Real = 1.0e-8)
    tol_weight = tol_weight_default(Δ)
    all(>(tol_weight), weights(Δ))::Bool || throw(
        ArgumentError("hybridization contains negligible or negative weight")
    )

    # Mean-field Green's function on eigenbasis
    H = arrowhead_matrix(Δ)
    H[1, 1] = ϵ_mf
    locs, V = eigen(Symmetric(H))
    amps = view(V, 1, :)

    # Split into occupied (valence) and unoccupied (conduction) sites.
    # Split weight at Fermi-level evenly.
    n_occ = count(<(-tol), locs)
    n_zero = count(ϵ -> abs(ϵ) <= tol, locs) # can be degenerate
    amp_zero = norm(view(amps, (n_occ + 1):(n_occ + n_zero)))
    locs_v = locs[1:n_occ]
    amps_v = amps[1:n_occ]
    locs_c = locs[(n_occ + n_zero + 1):end]
    amps_c = amps[(n_occ + n_zero + 1):end]
    if !iszero(amp_zero)
        push!(locs_v, 0)
        push!(amps_v, amp_zero / sqrt(2))
        pushfirst!(locs_c, 0)
        pushfirst!(amps_c, amp_zero / sqrt(2))
    end
    isempty(locs_v) && throw(ArgumentError("no occupied (valence) states"))
    isempty(locs_c) && throw(ArgumentError("no empty (conduction) states"))

    # tridiagonalization to get chains
    a_v = norm(amps_v)
    a_c = norm(amps_c)
    rmul!(amps_v, inv(a_v))
    rmul!(amps_c, inv(a_c))
    e_v, t_v = _tridiagonalize(locs_v, amps_v)
    e_c, t_c = _tridiagonalize(locs_c, amps_c)

    # Rotate bonding/antibonding orbitals back to impurity and mirror site.
    H_diag = SDiagonal(popfirst!(e_v), popfirst!(e_c))
    R = @SMatrix [
        a_v   a_c
        a_c  -a_v
    ]
    H_ib = R' * H_diag * R
    H_ib = (H_ib + H_ib') / 2

    # impurity-mirror to chain couplings
    T = eltype(H_ib)
    t_imp_v = isempty(t_v) ? zero(T) : popfirst!(t_v)
    t_imp_c = isempty(t_c) ? zero(T) : popfirst!(t_c)
    i_v = a_v * t_imp_v
    b_v = a_c * t_imp_v
    i_c = a_c * t_imp_c
    b_c = -a_v * t_imp_c

    return NaturalImpurityOrbital(H_ib, i_v, i_c, b_v, b_c, e_v, t_v, e_c, t_c)
end

"""
    natural_impurity_orbital_operator(
        H_nat::NaturalImpurityOrbital{T},
        H_int::Operator{T},
        ϵ_imp::T,
        fs::FockSpace,
        n_v_bit::Int = 1,
        n_c_bit::Int = 1,
    ) where {T <: Real}

Convert natural impurity orbital Hamiltonian to `Operator`.

Ordering is
`[i, b, n_v[1...n_v_bit], n_c[1...n_c_bit], n_v[n_v_bit+1...end], n_c[n_c_bit+1...end]`.

# Arguments
- `H_nat::NaturalImpurityOrbital{T}`: natural impurity orbital representation
- `H_int::Operator{T}`: interacting Hamiltonian
- `ϵ_imp::T`: on-site energy of impurity
- `fs::FockSpace`: Fock Space used for the system
- `n_v_bit::Int=1`: number of valence bath sites in bit component
- `n_c_bit::Int=1`: number of conduction bath sites in bit component
"""
function natural_impurity_orbital_operator(
        H_nat::NaturalImpurityOrbital{T},
        H_int::Operator{T},
        ϵ_imp::T,
        fs::FockSpace,
        n_v_bit::Int = 1,
        n_c_bit::Int = 1,
    ) where {T <: Real}
    n_v = n_valence(H_nat)
    n_c = n_conduction(H_nat)
    0 < n_v_bit <= n_v || throw(ArgumentError("too many valence sites in bit component"))
    0 < n_c_bit <= n_c || throw(ArgumentError("too many conduction sites in bit component"))
    c = annihilators(fs)
    n = occupations(fs)
    H = _add_impurity_terms(H_int, c, H_nat, ϵ_imp)
    for σ in axes(c, 2)
        # valence bath sites
        for i in 1:n_v
            if i <= n_v_bit
                j = 2 + i
            else
                j = 2 + i + n_c_bit
            end
            # bath site
            H += H_nat.e_v[i] * n[j, σ]
            if i == 1
                # hopping (i, b) ↔ v_1
                H += H_nat.i_v * c[1, σ]' * c[j, σ]
                H += H_nat.i_v * c[j, σ]' * c[1, σ]
                H += H_nat.b_v * c[2, σ]' * c[j, σ]
                H += H_nat.b_v * c[j, σ]' * c[2, σ]
            elseif i == n_v_bit + 1
                # hopping v_(n_v_bit) ↔ v_(n_v_bit + 1)
                H += H_nat.t_v[n_v_bit] * c[2 + n_v_bit, σ]' * c[j, σ]
                H += H_nat.t_v[n_v_bit] * c[j, σ]' * c[2 + n_v_bit, σ]
            else
                # hopping to previous neighbor
                H += H_nat.t_v[i - 1] * c[j - 1, σ]' * c[j, σ]
                H += H_nat.t_v[i - 1] * c[j, σ]' * c[j - 1, σ]
            end
        end
        # conduction bath sites
        for i in 1:n_c
            if i <= n_c_bit
                j = 2 + i + n_v_bit
            else
                j = 2 + i + n_v
            end
            # bath site
            H += H_nat.e_c[i] * n[j, σ]
            if i == 1
                # hopping (i, b) ↔ c_1
                H += H_nat.i_c * c[1, σ]' * c[j, σ]
                H += H_nat.i_c * c[j, σ]' * c[1, σ]
                H += H_nat.b_c * c[2, σ]' * c[j, σ]
                H += H_nat.b_c * c[j, σ]' * c[2, σ]
            elseif i == n_c_bit + 1
                # hopping c_(n_c_bit) ↔ c_(n_c_bit + 1)
                H += H_nat.t_c[n_c_bit] * c[2 + n_v_bit + n_c_bit, σ]' * c[j, σ]
                H += H_nat.t_c[n_c_bit] * c[j, σ]' * c[2 + n_v_bit + n_c_bit, σ]
            else
                # hopping to previous neighbor
                H += H_nat.t_c[i - 1] * c[j - 1, σ]' * c[j, σ]
                H += H_nat.t_c[i - 1] * c[j, σ]' * c[j - 1, σ]
            end
        end
    end
    return H
end

"""
    natural_impurity_orbital_ras_operator(
        H_nat::NaturalImpurityOrbital{T},
        H_int::Operator{T},
        ϵ_imp::T,
        fs::FockSpace,
        n_v_bit::Int = 1,
        n_c_bit::Int = 1,
        p::Int = 1,
    ) where {T <: Real}

Convert natural impurity orbital Hamiltonian to `RASOperator`.

# Arguments
- `H_nat::NaturalImpurityOrbital{T}`: natural impurity orbital representation
- `H_int::Operator{T}`: interacting Hamiltonian
- `ϵ_imp::T`: on-site energy of impurity
- `fs::FockSpace`: Fock Space used for the system
- `n_v_bit::Int=1`: number of valence bath sites in bit component
- `n_c_bit::Int=1`: number of conduction bath sites in bit component
- `p::Int=1`: maximum excitation in bit component

See also `RASOperator`.
"""
function natural_impurity_orbital_ras_operator(
        H_nat::NaturalImpurityOrbital{T},
        H_int::Operator{T},
        ϵ_imp::T,
        fs::FockSpace,
        n_v_bit::Int = 1,
        n_c_bit::Int = 1,
        p::Int = 1,
    ) where {T <: Real}
    p >= 0 || throw(ArgumentError("negative excitation"))
    if iszero(n_v_bit) && iszero(n_c_bit)
        return _natural_impurity_orbital_ras_operator_zero(
            H_nat, H_int, ϵ_imp, fs, p,
        )
    end
    n_v_bit >= 1 || throw(ArgumentError("invalid n_v_bit"))
    n_c_bit >= 1 || throw(ArgumentError("invalid n_c_bit"))
    n_v = n_valence(H_nat)
    n_c = n_conduction(H_nat)
    n_v_bit <= n_v || throw(ArgumentError("n_v_bit too big"))
    n_c_bit <= n_c || throw(ArgumentError("n_c_bit too big"))
    n_v_bit < n_v || n_c_bit < n_c || throw(
        ArgumentError("no restricted active space"),
    )

    # Create bit Operator
    n_bit = 2 + n_v_bit + n_c_bit
    c = annihilators(fs)
    H_bit = _add_impurity_terms(H_int, c, H_nat, ϵ_imp)
    for σ in axes(c, 2)
        H_bit = _add_chain_to_bit(H_bit, c, σ, H_nat, true, 2, n_v_bit)
        H_bit = _add_chain_to_bit(H_bit, c, σ, H_nat, false, 2 + n_v_bit, n_c_bit)
    end

    # Create VectorOperator
    n_v_vector = n_v - n_v_bit
    n_c_vector = n_c - n_c_bit
    esite = [H_nat.e_v[(1 + n_v_bit):end]; H_nat.e_c[(1 + n_c_bit):end]]
    # no hopping between valence/conduction chains
    separator = (n_v_vector > 0 && n_c_vector > 0) ? [zero(T)] : T[]
    ehop = [
        H_nat.t_v[(1 + n_v_bit):end];
        separator;
        H_nat.t_c[(1 + n_c_bit):end];
    ]

    # Create MixedOperator
    # (i, j, amp)
    mixed = ()
    if n_v_vector > 0
        mixed = (mixed..., (2 + n_v_bit, 1, H_nat.t_v[n_v_bit]))
    end
    if n_c_vector > 0
        mixed = (
            mixed...,
            (2 + n_v_bit + n_c_bit, n_v_vector + 1, H_nat.t_c[n_c_bit]),
        )
    end
    @assert !isempty(mixed) # necessary for JETLS, although guaranteed by earlier checks

    return RASOperator(H_bit, mixed, esite, ehop, n_bit, n_v_vector, n_c_vector, p)
end

# Add the on-site energies and nearest-neighbor hopping of given chain
# (valence or conduction) inside the RAS bit component.
function _add_chain_to_bit(
        H::Operator,
        c,
        σ,
        H_nat::NaturalImpurityOrbital,
        is_valence::Bool,
        offset_bit::Int,
        n_chain::Int,
    )
    e = is_valence ? H_nat.e_v : H_nat.e_c
    t = is_valence ? H_nat.t_v : H_nat.t_c
    for i in 1:n_chain
        site_bit = offset_bit + i
        # bath site
        H += e[i] * c[site_bit, σ]' * c[site_bit, σ]
        if i == 1
            # hopping chain start ↔ i and chain start ↔ b
            H_i = is_valence ? H_nat.i_v : H_nat.i_c
            H_b = is_valence ? H_nat.b_v : H_nat.b_c
            H += H_i * c[1, σ]' * c[site_bit, σ]
            H += H_i * c[site_bit, σ]' * c[1, σ]
            H += H_b * c[2, σ]' * c[site_bit, σ]
            H += H_b * c[site_bit, σ]' * c[2, σ]
        else
            # hopping to previous neighbor
            H += t[i - 1] * c[site_bit - 1, σ]' * c[site_bit, σ]
            H += t[i - 1] * c[site_bit, σ]' * c[site_bit - 1, σ]
        end
    end
    return H
end

# Add the impurity on-site energy and the impurity-mirror-site hopping.
# This block is identical for the full and RAS operator constructions.
function _add_impurity_terms(H, c, H_nat::NaturalImpurityOrbital, ϵ_imp)
    for σ in axes(c, 2)
        # impurity i
        H += ϵ_imp * c[1, σ]' * c[1, σ]
        # mirror site b
        H += H_nat.H_ib[2, 2] * c[2, σ]' * c[2, σ]
        # hopping i ↔ b
        H += H_nat.H_ib[2, 1] * c[1, σ]' * c[2, σ]
        H += H_nat.H_ib[2, 1] * c[2, σ]' * c[1, σ]
    end
    return H
end

# Same as `natural_impurity_orbital_operator` but with `n_v_bit === n_c_bit === 0`.
function _natural_impurity_orbital_ras_operator_zero(
        H_nat::NaturalImpurityOrbital{T},
        H_int::Operator{T},
        ϵ_imp::T,
        fs::FockSpace,
        p::Int,
    ) where {T <: Real}
    c = annihilators(fs)

    # Create bit Operator
    n_bit = 2
    H_bit = _add_impurity_terms(H_int, c, H_nat, ϵ_imp)

    # Create VectorOperator
    n_v_vector = n_valence(H_nat)
    n_c_vector = n_conduction(H_nat)
    esite = [H_nat.e_v; H_nat.e_c]
    # no hopping between valence/conduction chains
    separator = (n_v_vector > 0 && n_c_vector > 0) ? [zero(T)] : T[]
    ehop = [H_nat.t_v; separator; H_nat.t_c]

    # Create MixedOperator
    # (i, j, amp)
    # An empty chain has no site to hop to.
    mixed = ()
    if n_v_vector > 0
        mixed = (mixed..., (1, 1, H_nat.i_v), (2, 1, H_nat.b_v)) # i, b ↔ v1
    end
    if n_c_vector > 0
        mixed = (
            mixed...,
            (1, n_v_vector + 1, H_nat.i_c), # i ↔ c1
            (2, n_v_vector + 1, H_nat.b_c), # b ↔ c1
        )
    end
    # necessary for JETLS, although guaranteed by `natural_impurity_orbital`
    @assert !isempty(mixed)

    return RASOperator(H_bit, mixed, esite, ehop, n_bit, n_v_vector, n_c_vector, p)
end

# `v` must be normalized
function _tridiagonalize(locs::AbstractVector{<:Real}, v::AbstractVector{<:Real})
    Q = [v nullspace(v')]
    # Lower triangle preserves the first basis vector (impurity).
    _, _, α, β = LAPACK.hetrd!('L', Q' * Diagonal(locs) * Q)
    map!(abs, β)
    return α, β
end
