# Methods related to `Fermions.Wavefunctions` module.

# The mean-field ground state in the natural impurity orbital basis is Eq. (E8) of
# https://doi.org/10.1103/PhysRevB.90.085102,
#   (a_v i_↑^† + a_c b_↑^†) (a_v i_↓^† + a_c b_↓^†) on the filled valence sites,
# with a_v, a_c the impurity amplitudes in the occupied and empty subspace.
# It expands into four determinants:
# i doubly occupied (a_v^2), the singlet (a_v a_c, twice), and b doubly occupied (a_c^2).
# That state is exact for U = 0, but at finite U each double occupation costs U.
# For large U it needs more Krylov steps to converge.
# Hence the start keeps only the singlet and drops the double occupations.

"""
    Wavefunction_singlet(
        D::Type, L_v::Integer, L_c::Integer, V_v::Integer, V_c::Integer
    )

Initialize a `Wavefunction` with
singlet in impurity and mirror site
and all valence sites filled.
"""
function Wavefunction_singlet(
        D::Type, L_v::Integer, L_c::Integer, V_v::Integer, V_c::Integer
    )
    K, V = keytype(D), valtype(D)
    s1 = slater_start(K, 0b0110, L_v, L_c, V_v, V_c)
    s2 = slater_start(K, 0b1001, L_v, L_c, V_v, V_c)
    return Wavefunction(Dict(s1 => V(1 / sqrt(2)), s2 => V(1 / sqrt(2))))
end

"""
    RASWavefunction_singlet(
        D::Type, L_v::Integer, L_c::Integer, V_v::Integer, V_c::Integer, excitation::Integer
    )

Initialize a `RASWavefunction` with
singlet in impurity and mirror site
and all valence sites filled.
"""
function RASWavefunction_singlet(
        D::Type, L_v::Integer, L_c::Integer, V_v::Integer, V_c::Integer, excitation::Integer
    )
    K, V = keytype(D), valtype(D)
    s1 = slater_start(K, 0b0110, L_v, L_c, 0, 0)
    s2 = slater_start(K, 0b1001, L_v, L_c, 0, 0)
    v_dim = sum(i -> binomial(2 * (V_v + V_c), i), 0:excitation)
    vec = zeros(V, v_dim)
    vec[1] = 1 / sqrt(2)
    result = RASWavefunction(
        Dict(s1 => copy(vec), s2 => copy(vec)), 2 + L_v + L_c, V_v, V_c, excitation
    )
    return result
end

"""
    ground_state!(
        H::RASOperator,
        ψ_start::RASWavefunction,
        n_kryl::Integer,
        n_max_restart::Integer,
        variance::Real,
    )

Shift ``H → H - E_0`` in-place and return ``E_0``, ``|ψ_0⟩``.

Get approximate ground state and energy using steps of `n_kryl` Krylov cycles
and at most `n_max_restart` restarts.
Calculation is stopped early if `⟨H^2⟩ <= variance`.
"""
function ground_state!(
        H::RASOperator,
        ψ_start::RASWavefunction,
        n_kryl::Integer,
        n_max_restart::Integer,
        variance::Real,
    )
    # check input
    isapprox(norm(ψ_start), 1; atol = 10 * eps()) ||
        throw(ArgumentError("ψ_start is not normalized"))
    n_kryl >= 1 || throw(ArgumentError("n_kryl must be at least 1"))
    n_max_restart >= 1 || throw(ArgumentError("n_max_restart must be at least 1"))
    variance >= 0 || throw(ArgumentError("variance must be >= 0"))

    # initial guess
    # `H` is shifted to keep its spectrum near zero, so `E0` sums all shifts
    ψ0 = copy(ψ_start)
    E0 = dot(ψ0, H, ψ0)
    Fermions.shift_spectrum!(H, E0)

    # containers to reduce allocations
    a = Vector{Float64}(undef, n_kryl)
    b = Vector{Float64}(undef, n_kryl - 1)
    states = [similar(ψ_start) for _ in 1:(n_kryl + 1)]

    for itr in 1:n_max_restart
        lanczos!(a, b, states, H, ψ0, n_kryl)
        F = eigen!(SymTridiagonal(a, b))
        # new state is linear combination
        rmul!(ψ0, F.vectors[1, 1]) # rescale first element
        @inbounds for i in 2:n_kryl
            # add all other elements
            axpy!(F.vectors[i, 1], states[i], ψ0) # ψ0_new += c_i * ψ_i
        end
        normalize!(ψ0) # possible orthogonality loss in Lanczos
        E_shift = dot(ψ0, H, ψ0)
        Fermions.shift_spectrum!(H, E_shift)
        E0 += E_shift

        # calculate variance
        foo = H * ψ0
        var = foo ⋅ foo
        if var <= variance
            @debug "ground state variance reached after $(n_kryl * itr) Krylov steps"
            break
        elseif itr == n_max_restart
            @warn "target variance not reached, stopped at $(var)"
        end
    end

    return E0, ψ0
end
