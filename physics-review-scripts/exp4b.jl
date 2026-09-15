# Experiment 4b: zero-energy splitting and RAS accuracy; Lanczos-reached state = projection
using RAS_DMFT
using Fermions
using Fermions.Wavefunctions
using LinearAlgebra
using Logging
using Printf
include(joinpath(@__DIR__, "exp4_lib.jl"))

U = 2.0
ϵ_imp = -1.0
for (name, Δ) in (("PHS 4 poles (zero mode split)", PolesSum([-2.0, -1.0, 1.0, 2.0], fill(0.09, 4))),
                  ("PHS 5 poles (no zero mode)", PolesSum([-2.0, -1.0, 0.0, 1.0, 2.0], [0.09, 0.09, 0.1, 0.09, 0.09])),
                  ("PHS 6 poles (zero mode split)", PolesSum([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], fill(0.06, 6))),
                  ("PHS 7 poles (no zero mode)", PolesSum([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], [0.05, 0.05, 0.05, 0.06, 0.05, 0.05, 0.05])))
    println("=== $name ===")
    H_nat = natural_impurity_orbital(Δ, 0.0)
    n_v, n_c = n_valence(H_nat), n_conduction(H_nat)
    println("  basis size ", size(H_nat, 1), " vs star ", length(Δ) + 1, "; n_v=$n_v n_c=$n_c")
    # exact star energies per sector
    Hs, fss, ns = star_hamiltonian(Δ, U, ϵ_imp)
    Nhalf = ns  # half filling: ns electrons total
    Es = Dict{Int, Float64}()
    for N in (Nhalf - 1, Nhalf, Nhalf + 1)
        n1 = N ÷ 2; n2 = N - n1
        dets = sector_basis(ns, n1, n2)
        Es[N] = lowest_energy(Hs, dets)
        @printf("  star N=%d: E0=%.8f\n", N, Es[N])
    end
    Eglob = minimum(values(Es))
    # RAS
    for (L, p) in ((1, 1), (1, 2), (2, 1), (2, 2))
        (L < n_v && L < n_c) || continue
        fs = FockSpace(Orbitals(2 + 2L), FermionicSpin(1 // 2))
        n = occupations(fs)
        H_int = U * n[1, 1 // 2] * n[1, -1 // 2]
        H, E0, ψ0 = with_logger(NullLogger()) do
            init_system(Δ, H_int, ϵ_imp, 0.0, L, L, p, 1.0e-12)
        end
        @printf("  RAS L=%d p=%d: E0=%.8f  ΔE(global)=%.2e  n_imp=%.4f\n", L, p, E0, E0 - Eglob, dot(ψ0, n[1, 1 // 2] + n[1, -1 // 2], ψ0))
    end
end

# projection check for the 4-pole case
println("=== projection check, 4 poles ===")
Δ = PolesSum([-2.0, -1.0, 1.0, 2.0], fill(0.09, 4))
H_nat = natural_impurity_orbital(Δ, 0.0)
n_v, n_c = n_valence(H_nat), n_conduction(H_nat)
fs = FockSpace(Orbitals(2 + n_v + n_c), FermionicSpin(1 // 2))
n = occupations(fs); c = annihilators(fs)
H_int = U * n[1, 1 // 2] * n[1, -1 // 2]
Hn = natural_impurity_orbital_operator(H_nat, H_int, ϵ_imp, fs, n_v, n_c)
nsn = 2 + n_v + n_c
dets = sector_basis(nsn, n_v + 1, n_v + 1)
F = eigen(sector_matrix(Hn, dets))
ψs = Wavefunction_singlet(Dict{K, Float64}, n_v, n_c, 0, 0)
v = zeros(length(dets)); idx = Dict(d => i for (i, d) in enumerate(dets))
for (d, val) in pairs(ψs); v[idx[d]] = val; end
deg = findall(e -> abs(e - F.values[1]) < 1e-9, F.values)
println("  degenerate ground manifold size: ", length(deg))
Vd = F.vectors[:, deg]
ψ0 = normalize(Vd * (Vd' * v))
wf(ψ) = Wavefunction(Dict{K, Float64}(d => ψ[i] for (i, d) in enumerate(dets) if abs(ψ[i]) > 1e-15))
println("  impurity <n↑ - n↓> of projected state: ", dot(wf(ψ0), n[1, 1 // 2] - n[1, -1 // 2], wf(ψ0)))
G_nat = lehmann(Hn, fs, nsn, dets, ψ0, F.values[1], c[1, -1 // 2], n_v + 1, n_v + 1)
Hs, fss, ns = star_hamiltonian(Δ, U, ϵ_imp)
dets5 = sector_basis(ns, 2, 3)
F5 = eigen(sector_matrix(Hs, dets5))
cs = annihilators(fss)
G_dn = lehmann(Hs, fss, ns, dets5, F5.vectors[:, 1], F5.values[1], cs[1, -1 // 2], 2, 3)
G_up = lehmann(Hs, fss, ns, dets5, F5.vectors[:, 1], F5.values[1], cs[1, 1 // 2], 2, 3)
W = range(-4, 4; length = 801)
g_nat = evaluate_lorentzian(G_nat, W, 0.05)
g_avg = 0.5 .* (evaluate_lorentzian(G_dn, W, 0.05) .+ evaluate_lorentzian(G_up, W, 0.05))
@printf("  max|A_nat(projected start) - A_avg(doublet)| = %.2e (max A = %.3f)\n", maximum(abs, imag(g_nat) - imag(g_avg)), maximum(abs, imag(g_avg)))
# Also: actual Lanczos ground_state! result in the full bit space via RAS with L = n-1? Not possible (needs vector part).
