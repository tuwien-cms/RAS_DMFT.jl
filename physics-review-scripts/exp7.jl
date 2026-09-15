# Experiment 7: wrong-sector diagnostics and an S_z = 1/2 start state.
using RAS_DMFT
using Fermions
using Fermions.Wavefunctions
using LinearAlgebra
using Logging
using Printf
using SparseArrays
include(joinpath(@__DIR__, "exp4_lib.jl"))

function sector_matrix_sparse(H::Operator, dets)
    idx = Dict(d => i for (i, d) in enumerate(dets))
    I = Int[]; J = Int[]; V = Float64[]
    for (j, d) in enumerate(dets)
        ϕ = H * Wavefunction(Dict{K, Float64}(d => 1.0))
        for (det, val) in pairs(ϕ)
            push!(I, idx[det]); push!(J, j); push!(V, val)
        end
    end
    return sparse(I, J, V, length(dets), length(dets))
end
function lowest_sparse(M, nk = 150)
    n = size(M, 1)
    Q = zeros(n, nk); α = zeros(nk); β = zeros(nk - 1)
    q = normalize(ones(n) .+ 0.1 .* randn(n)); Q[:, 1] = q
    w = M * q; α[1] = dot(q, w); w -= α[1] * q; k = nk
    for j in 2:nk
        for _ in 1:2; w -= Q[:, 1:(j - 1)] * (Q[:, 1:(j - 1)]' * w); end
        β[j - 1] = norm(w); β[j - 1] < 1e-12 && (k = j - 1; break)
        q = w / β[j - 1]; Q[:, j] = q
        w = M * q - β[j - 1] * Q[:, j - 1]; α[j] = dot(q, w); w -= α[j] * q
    end
    return eigmin(SymTridiagonal(α[1:k], β[1:(k - 1)]))
end

Δ = hybridization_function_bethe_grid(range(-1, 1; length = 9))
Δ = PolesSum(locations(Δ) .+ 0.12, weights(Δ) .* [0.5, 0.7, 0.9, 1.0, 1.2, 1.3, 1.2, 1.0, 0.8])
U = 2.0
ϵ_imp = -0.4
ϵ_mf = 0.2223 # Hartree SC from exp2b → sector N=10 (E=-4.7397), global N=9 (E=-4.7592)
L = 1; p = 2
fs = FockSpace(Orbitals(2 + 2L), FermionicSpin(1 // 2))
c = annihilators(fs); n = occupations(fs)
H_int = U * n[1, 1 // 2] * n[1, -1 // 2]
d_dag = c[1, -1 // 2]'
H, E0, ψ0 = with_logger(NullLogger()) do
    init_system(Δ, H_int, ϵ_imp, ϵ_mf, L, L, p, 1.0e-12)
end
@printf("RAS N=10 sector: E0 = %.6f (exact N=10: -4.739672, exact N=9: -4.759178)\n", E0)
println("correlator_plus with d† (goes to N=11) and correlator_minus with d (goes to N=9):")
C_plus = correlator_plus(H, ψ0, d_dag, 60)
C_minus = correlator_minus(H, ψ0, d_dag', 60)
wneg = sum(w for (l, w) in C_plus if l < 0; init = 0.0)
wpos = sum(w for (l, w) in C_minus if l > 0; init = 0.0)
@printf("  C+ weight at ω<0: %.4f ; C- weight at ω>0: %.4f ; lowest C- pole location: %.4f (= -(E(N=9)-E0) → %.4f expected)\n",
    wneg, wpos, maximum(locations(C_minus)), -(-4.759178 - E0))

# S_z = 1/2 start: bit pattern 0b0111 → i↑, b↑, i↓ ; valence chain filled → N = 2 n_v + 3? no: 2 n_v + 3 electrons
H_nat = natural_impurity_orbital(Δ, ϵ_mf)
n_v, n_c = n_valence(H_nat), n_conduction(H_nat)
println("n_v = $n_v → singlet start has N = $(2 * (n_v + 1)); S_z=1/2 start with 0b0111 has N = $(2 * n_v + 3)")
# want N = 9 = 2 n_v + 1 → remove one electron from the pair: pattern 0b0001 (i↓ only) gives N = 2 n_v + 1
Kt = UInt64
s = slater_start(Kt, 0b0001, L, L, 0, 0)
v_dim = sum(i -> binomial(2 * (H.nfilled + H.nempty), i), 0:p)
vec = zeros(v_dim); vec[1] = 1.0
ψ_half = RASWavefunction(Dict(s => vec), 2 + 2L, H.nfilled, H.nempty, p)
E_half, ψ_h = with_logger(NullLogger()) do
    Hh = natural_impurity_orbital_ras_operator(H_nat, H_int, ϵ_imp, fs, L, L, p)
    ground_state!(Hh, ψ_half, 5, typemax(Int), 1.0e-12)
end
@printf("RAS from S_z=1/2 start (N=%d): E0 = %.6f (exact N=9: -4.759178)  n_imp=%.4f\n", 2n_v + 1, E_half, dot(ψ_h, n[1, 1 // 2] + n[1, -1 // 2], ψ_h))
