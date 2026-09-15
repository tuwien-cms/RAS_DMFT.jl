# Experiment 4: zero-energy splitting. PHS bath with even number of poles → reference has a
# zero eigenvalue → natural orbital basis has N+2 orbitals. What ground state is found?
using RAS_DMFT
using Fermions
using Fermions.Wavefunctions
using LinearAlgebra
using Printf

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

# Lehmann G for operator d (annihilator) given ground state vector in sector dets0
function lehmann(H, fs, nsites, dets0, ψ0, E0, d, Ns1, Ns2)
    ϕ = d' * Wavefunction(Dict{K, Float64}(K(0) => 1.0))
    det1 = first(keys(ϕ))
    mask = (K(1) << nsites) - 1
    lowblock = count_ones(det1 & mask) == 1
    dets_p = lowblock ? sector_basis(nsites, Ns1 + 1, Ns2) : sector_basis(nsites, Ns1, Ns2 + 1)
    dets_m = lowblock ? sector_basis(nsites, Ns1 - 1, Ns2) : sector_basis(nsites, Ns1, Ns2 - 1)
    idx_p = Dict(dd => i for (i, dd) in enumerate(dets_p))
    idx_m = Dict(dd => i for (i, dd) in enumerate(dets_m))
    Fp = eigen(sector_matrix(H, dets_p))
    Fm = eigen(sector_matrix(H, dets_m))
    vp = apply(d', ψ0, dets0, idx_p)
    vm = apply(d, ψ0, dets0, idx_m)
    wp = abs2.(Fp.vectors' * vp); lp = Fp.values .- E0
    wm = abs2.(Fm.vectors' * vm); lm = -(Fm.values .- E0)
    keep_p = wp .> 1.0e-13; keep_m = wm .> 1.0e-13
    return PolesSum(vcat(lp[keep_p], lm[keep_m]), vcat(wp[keep_p], wm[keep_m]))
end

# PHS insulator-like bath with 4 poles (even) → zero eigenvalue in the reference
Δ = PolesSum([-2.0, -1.0, 1.0, 2.0], fill(0.09, 4))
U = 2.0
ϵ_imp = -1.0
H_nat = natural_impurity_orbital(Δ, 0.0)
println("natural orbital basis size: ", size(H_nat, 1), " (star: ", length(Δ) + 1, ")")
println("eigenvalues natural: ", round.(eigvals(Symmetric(Matrix(H_nat))); digits = 6))
println("eigenvalues star:    ", round.(eigvals(Symmetric(arrowhead_matrix(Δ))); digits = 6))

# exact star: sector energies
Hs, fss, ns = star_hamiltonian(Δ, U, ϵ_imp)
for (n1, n2) in ((2, 2), (2, 3), (3, 3), (3, 2))
    dets = sector_basis(ns, n1, n2)
    F = eigen(sector_matrix(Hs, dets))
    @printf("star sector (%d,%d) N=%d: E0=%.8f E1=%.8f\n", n1, n2, n1 + n2, F.values[1], F.values[2])
end

# natural orbital basis: full operator with everything in bit part, sector N = 2(n_v+1), S_z=0
n_v, n_c = n_valence(H_nat), n_conduction(H_nat)
fs = FockSpace(Orbitals(2 + n_v + n_c), FermionicSpin(1 // 2))
n = occupations(fs)
c = annihilators(fs)
H_int = U * n[1, 1 // 2] * n[1, -1 // 2]
Hn = natural_impurity_orbital_operator(H_nat, H_int, ϵ_imp, fs, n_v, n_c)
nsn = 2 + n_v + n_c
dets = sector_basis(nsn, n_v + 1, n_v + 1)
F = eigen(sector_matrix(Hn, dets))
@printf("natural basis sector N=%d S_z=0: E0=%.8f E1=%.8f E2=%.8f\n", 2 * (n_v + 1), F.values[1], F.values[2], F.values[3])
# state reached from singlet start
ψs = Wavefunction_singlet(Dict{K, Float64}, n_v, n_c, 0, 0)
v = zeros(length(dets))
idx = Dict(d => i for (i, d) in enumerate(dets))
for (d, val) in pairs(ψs)
    v[idx[d]] = val
end
ov = abs.(F.vectors' * v)
i0 = findfirst(>(1e-8), ov)
@printf("lowest state with overlap to singlet start: E=%.8f (index %d), overlaps of degenerate manifold: %s\n", F.values[i0], i0, string(round.(ov[1:4]; digits = 4)))
ψ0 = F.vectors[:, i0]
# impurity spin-resolved occupation and S² in the reached state
Sz_imp = dot(Wavefunction(Dict{K, Float64}(d => ψ0[i] for (i, d) in enumerate(dets))), n[1, 1 // 2] - n[1, -1 // 2], Wavefunction(Dict{K, Float64}(d => ψ0[i] for (i, d) in enumerate(dets))))
println("impurity <n↑ - n↓> in reached state: ", Sz_imp)
# G_↓ in the natural basis (exact within enlarged space)
G_nat = lehmann(Hn, fs, nsn, dets, ψ0, F.values[i0], c[1, -1 // 2], n_v + 1, n_v + 1)
# exact star: doublet N=5 ground state, spin-averaged G
dets5 = sector_basis(ns, 2, 3)
F5 = eigen(sector_matrix(Hs, dets5))
cs = annihilators(fss)
G_dn = lehmann(Hs, fss, ns, dets5, F5.vectors[:, 1], F5.values[1], cs[1, -1 // 2], 2, 3)
G_up = lehmann(Hs, fss, ns, dets5, F5.vectors[:, 1], F5.values[1], cs[1, 1 // 2], 2, 3)
W = range(-4, 4; length = 801)
g_nat = evaluate_lorentzian(G_nat, W, 0.05)
g_avg = 0.5 .* (evaluate_lorentzian(G_dn, W, 0.05) .+ evaluate_lorentzian(G_up, W, 0.05))
g_dn = evaluate_lorentzian(G_dn, W, 0.05)
@printf("max|A_nat - A_avg(doublet)| = %.2e, max|A_nat - A_dn(doublet member)| = %.2e, max A = %.3f\n",
    maximum(abs, imag(g_nat) - imag(g_avg)), maximum(abs, imag(g_nat) - imag(g_dn)), maximum(abs, imag(g_avg)))
# also compare against N=4 and N=6 singlet ground states of the star
for (n1, n2) in ((2, 2), (3, 3))
    d4 = sector_basis(ns, n1, n2)
    F4 = eigen(sector_matrix(Hs, d4))
    G4 = lehmann(Hs, fss, ns, d4, F4.vectors[:, 1], F4.values[1], cs[1, -1 // 2], n1, n2)
    @printf("max|A_nat - A(N=%d singlet)| = %.2e\n", n1 + n2, maximum(abs, imag(g_nat) - imag(evaluate_lorentzian(G4, W, 0.05))))
end
