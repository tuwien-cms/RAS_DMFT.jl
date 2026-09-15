# Experiment 2: RAS truncation error as a function of ϵ_mf, fixed sector.
# Reference: exact diagonalization in the (N_dn, N_up) sector of the full model.
using RAS_DMFT
using Fermions
using Fermions.Wavefunctions
using LinearAlgebra
using Printf
using SparseArrays

const K = UInt64

# basis of determinants with n1 electrons in the low block, n2 in the high block
function sector_basis(nsites, n1, n2)
    dets = K[]
    for k in 0:(2^(2 * nsites) - 1)
        mask = (K(1) << nsites) - 1
        count_ones(K(k) & mask) == n1 || continue
        count_ones((K(k) >> nsites) & mask) == n2 || continue
        push!(dets, K(k))
    end
    return dets
end

function sector_matrix(H::Operator, dets)
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

function exact(Δ, U, ϵ_imp, N_per_spin)
    # basis independent of ϵ_mf: use plain arrowhead (star) Hamiltonian
    n_b = length(Δ)
    nsites = 1 + n_b
    fs = FockSpace(Orbitals(nsites), FermionicSpin(1 // 2))
    c = annihilators(fs)
    n = occupations(fs)
    H = U * n[1, 1 // 2] * n[1, -1 // 2]
    A = arrowhead_matrix(Δ)
    for σ in (-1 // 2, 1 // 2)
        H += ϵ_imp * n[1, σ]
        for k in 1:n_b
            H += A[k + 1, k + 1] * n[k + 1, σ]
            H += A[1, k + 1] * (c[1, σ]' * c[k + 1, σ] + c[k + 1, σ]' * c[1, σ])
        end
    end
    dets = sector_basis(nsites, N_per_spin, N_per_spin)
    M = sector_matrix(H, dets)
    F = eigen(Symmetric(Matrix(M)))
    E0 = F.values[1]
    v = F.vectors[:, 1]
    ψ = Wavefunction(Dict{K, Float64}(d => v[i] for (i, d) in enumerate(dets) if abs(v[i]) > 1.0e-14))
    nimp = dot(ψ, n[1, 1 // 2] + n[1, -1 // 2], ψ)
    docc = dot(ψ, n[1, 1 // 2] * n[1, -1 // 2], ψ)
    return E0, nimp, docc, F.values[2] - E0
end

function ras(Δ, U, ϵ_imp, ϵ_mf, L_v, L_c, p)
    fs = FockSpace(Orbitals(2 + L_v + L_c), FermionicSpin(1 // 2))
    n = occupations(fs)
    H_int = U * n[1, 1 // 2] * n[1, -1 // 2]
    H, E0, ψ0 = with_logger(NullLogger()) do
        init_system(Δ, H_int, ϵ_imp, ϵ_mf, L_v, L_c, p, 1.0e-12)
    end
    nimp = dot(ψ0, n[1, 1 // 2] + n[1, -1 // 2], ψ0)
    docc = dot(ψ0, n[1, 1 // 2] * n[1, -1 // 2], ψ0)
    H_nat = natural_impurity_orbital(Δ, ϵ_mf)
    return E0, nimp, docc, n_valence(H_nat)
end

using Logging

# bath: 7 poles, asymmetric semicircle-like weights
Δ = hybridization_function_bethe_grid(range(-1, 1; length = 7))
# make it asymmetric: shift and rescale weights a bit
Δ = PolesSum(locations(Δ) .+ 0.15, weights(Δ) .* [0.6, 0.8, 1.0, 1.2, 1.3, 1.2, 0.9])
println("Δ: ", length(Δ), " poles, moment0 = ", moment(Δ, 0))
U = 2.0
for ϵ_imp in (-1.0, -0.4, -1.6)
    println("=== U=$U ϵ_imp=$ϵ_imp ===")
    # candidate ϵ_mf values
    cands = [("bare ϵ_imp", ϵ_imp), ("ϵ_imp+U/2", ϵ_imp + U / 2)]
    # Hartree self-consistent: ϵ_mf = ϵ_imp + U n_σ^{MF}
    ϵ = ϵ_imp + U / 2
    for _ in 1:200
        A = arrowhead_matrix(Δ); A[1, 1] = ϵ
        F = eigen(Symmetric(A))
        nσ = sum(abs2, F.vectors[1, F.values .< 0])
        ϵ = 0.5 * ϵ + 0.5 * (ϵ_imp + U * nσ)
    end
    push!(cands, ("Hartree SC", ϵ))
    # sector from each candidate
    for (name, ϵ_mf) in cands
        H_nat = natural_impurity_orbital(Δ, ϵ_mf)
        Ns = n_valence(H_nat) + 1
        Eex, nex, dex, gap = exact(Δ, U, ϵ_imp, Ns)
        @printf("%-12s ϵ_mf=%+.4f sector N=%d | exact: E=%.6f n=%.4f d=%.4f gap=%.3f\n", name, ϵ_mf, 2Ns, Eex, nex, dex, gap)
        for (L, p) in ((1, 1), (1, 2), (2, 1), (2, 2))
            E0, nimp, docc, _ = ras(Δ, U, ϵ_imp, ϵ_mf, L, L, p)
            @printf("    RAS L=%d p=%d: ΔE=%.2e Δn=%+.2e Δd=%+.2e\n", L, p, E0 - Eex, nimp - nex, docc - dex)
        end
    end
    # also: ϵ_mf tuned so that mean-field n matches exact n of the Hartree-SC sector
    (name, ϵ_mf) = cands[3]
    H_nat = natural_impurity_orbital(Δ, ϵ_mf)
    Ns = n_valence(H_nat) + 1
    Eex, nex, dex, gap = exact(Δ, U, ϵ_imp, Ns)
    # bisection on ϵ_mf to match nex (per spin) within the same sector
    f(ϵ) = begin
        A = arrowhead_matrix(Δ); A[1, 1] = ϵ
        F = eigen(Symmetric(A))
        2 * sum(abs2, F.vectors[1, F.values .< 0]) - nex
    end
    lo, hi = ϵ_mf - 3, ϵ_mf + 3
    for _ in 1:60
        mid = (lo + hi) / 2
        f(mid) > 0 ? (lo = mid) : (hi = mid)
    end
    ϵ_occ = (lo + hi) / 2
    H_nat2 = natural_impurity_orbital(Δ, ϵ_occ)
    @printf("%-12s ϵ_mf=%+.4f sector N=%d (n_mf matches exact n=%.4f)\n", "occupation", ϵ_occ, 2 * (n_valence(H_nat2) + 1), nex)
    if n_valence(H_nat2) + 1 == Ns
        for (L, p) in ((1, 1), (1, 2), (2, 1), (2, 2))
            E0, nimp, docc, _ = ras(Δ, U, ϵ_imp, ϵ_occ, L, L, p)
            @printf("    RAS L=%d p=%d: ΔE=%.2e Δn=%+.2e Δd=%+.2e\n", L, p, E0 - Eex, nimp - nex, docc - dex)
        end
    end
end
