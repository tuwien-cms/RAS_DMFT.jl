# Experiment 3: impurity Green's function and self-energy of RAS vs exact Lehmann,
# 7-pole bath, effect of ϵ_mf. Also moment sum rules.
using RAS_DMFT
using Fermions
using Fermions.Wavefunctions
using LinearAlgebra
using Logging
using Printf
using SparseArrays

const K = UInt64

function sector_basis(nsites, n1, n2)
    dets = K[]
    mask = (K(1) << nsites) - 1
    for k in 0:(2^(2 * nsites) - 1)
        count_ones(K(k) & mask) == n1 || continue
        count_ones((K(k) >> nsites) & mask) == n2 || continue
        push!(dets, K(k))
    end
    return dets
end

function sector_matrix(H::Operator, dets)
    idx = Dict(d => i for (i, d) in enumerate(dets))
    M = zeros(length(dets), length(dets))
    for (j, d) in enumerate(dets)
        ϕ = H * Wavefunction(Dict{K, Float64}(d => 1.0))
        for (det, val) in pairs(ϕ)
            M[idx[det], j] += val
        end
    end
    return Symmetric(M)
end

function star_hamiltonian(Δ, U, ϵ_imp)
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
    return H, fs, nsites
end

# apply Operator to vector in basis dets_in, return vector in basis dets_out
function apply(O::Operator, v, dets_in, idx_out)
    w = zeros(length(idx_out))
    for (i, d) in enumerate(dets_in)
        abs(v[i]) < 1.0e-15 && continue
        ϕ = O * Wavefunction(Dict{K, Float64}(d => v[i]))
        for (det, val) in pairs(ϕ)
            w[idx_out[det]] += val
        end
    end
    return w
end

# exact Lehmann G_↓ for ground state in sector (N,N); which spin is "low block"? use both ops.
function exact_G(Δ, U, ϵ_imp, Ns)
    H, fs, nsites = star_hamiltonian(Δ, U, ϵ_imp)
    c = annihilators(fs)
    d = c[1, -1 // 2]
    dets0 = sector_basis(nsites, Ns, Ns)
    F0 = eigen(sector_matrix(H, dets0))
    E0 = F0.values[1]; ψ0 = F0.vectors[:, 1]
    # find which block spin -1//2 changes: apply d† and check counts
    ϕ = d' * Wavefunction(Dict{K, Float64}(K(0) => 1.0))
    det1 = first(keys(ϕ))
    mask = (K(1) << nsites) - 1
    lowblock = count_ones(det1 & mask) == 1
    dets_p = lowblock ? sector_basis(nsites, Ns + 1, Ns) : sector_basis(nsites, Ns, Ns + 1)
    dets_m = lowblock ? sector_basis(nsites, Ns - 1, Ns) : sector_basis(nsites, Ns, Ns - 1)
    idx_p = Dict(dd => i for (i, dd) in enumerate(dets_p))
    idx_m = Dict(dd => i for (i, dd) in enumerate(dets_m))
    Fp = eigen(sector_matrix(H, dets_p))
    Fm = eigen(sector_matrix(H, dets_m))
    vp = apply(d', ψ0, dets0, idx_p)
    vm = apply(d, ψ0, dets0, idx_m)
    wp = abs2.(Fp.vectors' * vp); lp = Fp.values .- E0
    wm = abs2.(Fm.vectors' * vm); lm = -(Fm.values .- E0)
    keep_p = wp .> 1.0e-14; keep_m = wm .> 1.0e-14
    G = PolesSum(vcat(lp[keep_p], lm[keep_m]), vcat(wp[keep_p], wm[keep_m]))
    return G, E0
end

function ras_G(Δ, U, ϵ_imp, ϵ_mf, L, p, n_kryl)
    fs = FockSpace(Orbitals(2 + L + L), FermionicSpin(1 // 2))
    c = annihilators(fs)
    n = occupations(fs)
    H_int = U * n[1, 1 // 2] * n[1, -1 // 2]
    d_dag = c[1, -1 // 2]'
    q_dag = H_int * d_dag - d_dag * H_int
    H, E0, ψ0 = with_logger(NullLogger()) do
        init_system(Δ, H_int, ϵ_imp, ϵ_mf, L, L, p, 1.0e-12)
    end
    Σ_H = dot(ψ0, q_dag' * d_dag + d_dag * q_dag', ψ0)
    O = [q_dag - Σ_H * d_dag, d_dag]
    C_plus, C_minus = with_logger(NullLogger()) do
        correlator_plus(H, ψ0, O, n_kryl), correlator_minus(H, ψ0, map(adjoint, O), n_kryl)
    end
    C = transpose(C_minus) + C_plus
    merge_small_weight!(C, 1.0e-12)
    G = PolesSum(C, 2, 2)
    Σ = PolesSum(self_energy_IFG(C, 1), 1, 1)
    nup = dot(ψ0, n[1, 1 // 2], ψ0)
    return G, Σ, Σ_H, E0, nup
end

Δ = hybridization_function_bethe_grid(range(-1, 1; length = 7))
Δ = PolesSum(locations(Δ) .+ 0.15, weights(Δ) .* [0.6, 0.8, 1.0, 1.2, 1.3, 1.2, 0.9])
U = 2.0
W = range(-4, 4; length = 801)
δ = 0.1
for ϵ_imp in (-1.0, -0.4)
    println("=== U=$U ϵ_imp=$ϵ_imp ===")
    ϵ_hf = let ϵ = ϵ_imp + U / 2
        for _ in 1:300
            A = arrowhead_matrix(Δ); A[1, 1] = ϵ
            F = eigen(Symmetric(A))
            nσ = sum(abs2, F.vectors[1, F.values .< 0])
            ϵ = 0.7 * ϵ + 0.3 * (ϵ_imp + U * nσ)
        end
        ϵ
    end
    for (name, ϵ_mf) in (("bare", ϵ_imp), ("Hartree SC", ϵ_hf))
        H_nat = natural_impurity_orbital(Δ, ϵ_mf)
        Ns = n_valence(H_nat) + 1
        Gex, E0ex = exact_G(Δ, U, ϵ_imp, Ns)
        gex = evaluate_lorentzian(Gex, W, δ)
        # exact self-energy through Dyson on the complex contour
        g0inv = W .+ im * δ .- ϵ_imp .- evaluate_lorentzian(Δ, W, δ)
        sex = g0inv .- 1 ./ gex
        @printf("  %-10s ϵ_mf=%+.4f sector N=%d | exact: M0=%.6f M1=%.6f M2=%.6f\n", name, ϵ_mf, 2Ns, moment(Gex, 0), moment(Gex, 1), moment(Gex, 2))
        for (L, p, nk) in ((1, 1, 40), (1, 2, 60), (2, 2, 60))
            (L < n_valence(H_nat) && L < n_conduction(H_nat)) || continue
            G, Σ, Σ_H, E0, nup = ras_G(Δ, U, ϵ_imp, ϵ_mf, L, p, nk)
            g = evaluate_lorentzian(G, W, δ)
            s = evaluate_lorentzian(Σ, W, δ) .+ Σ_H
            errA = maximum(abs, imag(g) - imag(gex)) / maximum(abs, imag(gex))
            errS = maximum(abs, imag(s) - imag(sex)) / maximum(abs, imag(sex))
            errSr = maximum(abs, real(s) - real(sex)) / maximum(abs, real(sex))
            m2_expected = ϵ_imp^2 + 2 * ϵ_imp * Σ_H + U^2 * nup + moment(Δ, 0)
            @printf("      RAS L=%d p=%d: ΔE0=%.1e | G: M0-1=%.1e M1-(ϵ+Σ_H)=%.1e M2-th=%.1e | rel.max err A=%.2e ImΣ=%.2e ReΣ=%.2e | Σ_dyn M0-U²n(1-n)=%.1e | wrong-sign poles G: %d\n",
                L, p, E0 - E0ex, moment(G, 0) - 1, moment(G, 1) - (ϵ_imp + Σ_H), moment(G, 2) - m2_expected,
                errA, errS, errSr, moment(Σ, 0) - U^2 * nup * (1 - nup),
                0)
        end
    end
end
