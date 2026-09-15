# utility functions

function _with_blas_threads(f, t::Int = 1)
    t_old = BLAS.get_num_threads()
    t_old == t && return f()

    BLAS.set_num_threads(t)
    try
        return f()
    finally
        BLAS.set_num_threads(t_old)
    end
end

"""
    get_RAS_parameters(n_sites::Int, n_v::Int, n_v_bit::Int, n_c_bit::Int)

Return `n_bit`, `n_v_vector`, `n_c_vector`.
"""
function get_RAS_parameters(n_sites::Int, n_v::Int, n_v_bit::Int, n_c_bit::Int)
    n_bit = 2 + n_v_bit + n_c_bit
    n_c = n_sites - n_v - 2
    n_v_vector = n_v - n_v_bit
    n_c_vector = n_c - n_c_bit
    return n_bit, n_v_vector, n_c_vector
end

"""
    init_system(
        Δ::PolesSum,
        H_int::Operator,
        ϵ_imp::Real,
        ϵ_mf::Real,
        L_v::Int,
        L_c::Int,
        p::Int,
        var::Real,
    )

Return Hamiltonian, ground state energy, and ground state.

`ϵ_imp` is the bare impurity level entering the Hamiltonian, `ϵ_mf` the mean-field
level defining the basis, see [`natural_impurity_orbital`](@ref).
"""
function init_system(
        Δ::PolesSum,
        H_int::Operator,
        ϵ_imp::Real,
        ϵ_mf::Real,
        L_v::Int,
        L_c::Int,
        p::Int,
        var::Real,
    )
    H_nat = natural_impurity_orbital(Δ, ϵ_mf)
    n_bit, V_v, V_c = get_RAS_parameters(size(H_nat, 1), n_valence(H_nat), L_c, L_v)
    fs = FockSpace(Orbitals(n_bit), FermionicSpin(1 // 2))
    H = natural_impurity_orbital_ras_operator(H_nat, H_int, ϵ_imp, fs, L_v, L_c, p)
    ψ_start = RASWavefunction_singlet(Dict{UInt64, Float64}, L_v, L_c, V_v, V_c, p)
    E0, ψ0 = ground_state!(H, ψ_start, 5, typemax(Int), var)
    return H, E0, ψ0
end

"""
    temperature_kondo(U::Real, ϵ::Real, Δ0::Real)

Calculate the Kondo temperature for
an interaction `U`,
on-site with energy `ϵ`,
and hybridization `Δ0`.

```math
T_\\mathrm{K} = \\sqrt{\\frac{UΔ_0}{2}} \\exp(\\frac{π ϵ(ϵ+U)}{2UΔ_0})
```
"""
function temperature_kondo(U::Real, ϵ::Real, Δ0::Real)
    return sqrt(U * Δ0 / 2) * exp(π * ϵ * (ϵ + U) / (2 * U * Δ0))
end

"""
    find_chemical_potential(
        H_k::Vector{<:AbstractMatrix},
        Σ_stat::AbstractMatrix,
        Σ_dyn::PolesSumBlock,
        n_fill::Real;
        μ_tol::Real = 1.0e-6,
        b_max::Int = 30,
        μ_min::Real = minimum(locations(Σ_dyn)),
        μ_max::Real = maximum(locations(Σ_dyn)),
        tol_weight::Real = 0,
    )

Find chemical potential ``μ``, such that desired filling ``n_\\mathrm{fill}`` is fulfilled

```math
\\begin{aligned}
n_\\mathrm{fill}
& ≡
∫_{-∞}^0 \\mathrm{d}ω~\\mathrm{Tr}
\\left[
-\\frac{1}{π}\\mathrm{Im}~G_\\mathrm{loc}(ω+\\mathrm{i}0^+)
\\right] \\\\
& =
∫_{-∞}^0 \\mathrm{d}ω~\\mathrm{Tr}
\\left[
-\\frac{1}{π}\\mathrm{Im}~
\\frac{1}{N_k} ∑_k \\frac{1}{ω + \\mathrm{i}0^+ +μ - H_k - Σ(ω + \\mathrm{i}0^+)}
\\right] .
\\end{aligned}
```

A bisection algorithm is used which stops once `Δμ < μ_tol`
or `b_max` iterations are surpassed.

Returns the calculated chemical potential and effective filling.

# Arguments
- `μ_tol::Real = 1.0e-6`: tolerance `Δμ` to exit bisection early
- `b_max::Int = 30`: maximum number of bisections
- `μ_min::Real = minimum(locations(Σ_dyn))`: initial lower bound for `μ`
- `μ_max::Real = maximum(locations(Σ_dyn))`: initial upper bound for `μ`
- `tol_weight::Real = 0`: treat weights less or equal than this value in `Σ_dyn` as zero
"""
function find_chemical_potential(
        H_k::Vector{<:AbstractMatrix},
        Σ_stat::AbstractMatrix,
        Σ_dyn::PolesSumBlock,
        n_fill::Real;
        μ_tol::Real = 1.0e-6,
        b_max::Int = 30,
        μ_min::Real = minimum(locations(Σ_dyn)),
        μ_max::Real = maximum(locations(Σ_dyn)),
        tol_weight::Real = 0,
    )
    # check input
    n_b = size(first(H_k), 1)
    allequal(size, H_k)::Bool || throw(DimensionMismatch("different matrix sizes in H_k"))
    size(Σ_stat) == (n_b, n_b) || throw(DimensionMismatch("size of Σ_stat does not match H_k"))
    (size(Σ_dyn) == (n_b, n_b))::Bool || throw(DimensionMismatch("size of Σ_dyn does not match H_k"))
    μ_min < μ_max || throw(ArgumentError("violating μ_min < μ_max"))

    # represent dynamic part of self-energy as block arrowhead matrix
    Σ_A = arrowhead_matrix(Σ_dyn, sqrt(tol_weight); thin = true)

    # filling for initial guesses
    n_min = _filling_mu(H_k, Σ_stat, Σ_A, μ_min)
    n_max = _filling_mu(H_k, Σ_stat, Σ_A, μ_max)
    n_min <= n_fill <= n_max ||
        throw(ArgumentError("violating n(μ_min) = $(n_min) <= n_fill <= n(μ_max) = $(n_max)"))

    # bisect chemical potential μ
    μ_new = 0.0
    n_new = 0.0
    n_bisect = 0
    for _ in 1:b_max
        n_bisect += 1
        μ_new = 0.5 * (μ_min + μ_max)
        n_new = _filling_mu(H_k, Σ_stat, Σ_A, μ_new)
        n_new > n_fill ? μ_max = μ_new : μ_min = μ_new
        (μ_max - μ_min) < μ_tol && break
    end
    @debug "chemical potential bisection" n_bisect μ_new μ_min μ_max n_fill n_new

    return μ_new, n_new
end

# Diagonalize `Σ_A` with its top-left `n_b × n_b` block replaced by `H + Σ_stat - μ * I`.
function _arrowhead_eigen(
        Σ_A::AbstractMatrix,
        H::AbstractMatrix,
        Σ_stat::AbstractMatrix,
        μ::Real,
        n_b::Int,
    )
    foo = copy(Σ_A)
    view(foo, 1:n_b, 1:n_b) .= H .+ Σ_stat .- μ * one(H) # one fused broadcast, no temporaries
    return eigen!(Hermitian(foo))
end

# Calculate filling for given chemical potential μ.
function _filling_mu(H_k, Σ_stat, Σ_A::AbstractMatrix, μ)
    n_b = LinearAlgebra.checksquare(first(H_k)) # number of bands
    z = zero(float(real(eltype(Σ_A))))
    result = Threads.Atomic{typeof(z)}(z)

    Threads.@threads for i in eachindex(H_k)
        # NOTE: `Σ_A` is a sparse (block arrowhead matrix).
        # One can use Krylov methods to approximate spectrum
        # if full decomposition is too slow.
        F = _arrowhead_eigen(Σ_A, H_k[i], Σ_stat, μ, n_b)
        n_loc = z # local filling
        @inbounds for j in axes(Σ_A, 2)
            ϵ = F.values[j]
            v = @view F.vectors[1:n_b, j]
            if ϵ < 0
                # Trace of v*v' is sum of values squared.
                n_loc += sum(abs2, v)
            end
        end
        Threads.atomic_add!(result, n_loc)
    end

    return result[] /= length(H_k)
end

function _issorted_and_unique(grid::AbstractVector{<:Real})
    issorted(grid) || throw(ArgumentError("grid is not sorted"))
    allunique(grid) || throw(ArgumentError("grid has degenerate locations"))
    # isequal() treats -0.0 and 0.0 as unequal although both are zero.
    count(iszero, grid) <= 1 || throw(ArgumentError("grid has duplicate zeros"))
    return true
end
