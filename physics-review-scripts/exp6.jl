# Experiment 6: asymmetric bit sizes L_v ≠ L_c; analytic Bethe branch; broadening formula.
using RAS_DMFT
using Fermions
using Fermions.Wavefunctions
using LinearAlgebra
using Logging
using Printf
include(joinpath(@__DIR__, "exp4_lib.jl"))

U = 2.0
ϵ_imp = -0.7
Δ = PolesSum([-2.0, -1.0, 0.3, 1.0, 2.0], [0.09, 0.09, 0.1, 0.09, 0.09])
ϵ_mf = 0.2
H_nat = natural_impurity_orbital(Δ, ϵ_mf)
n_v, n_c = n_valence(H_nat), n_conduction(H_nat)
println("n_v=$n_v n_c=$n_c size=", size(H_nat, 1))
nsites = 2 + n_v + n_c
fs = FockSpace(Orbitals(nsites), FermionicSpin(1 // 2))
n = occupations(fs)
H_int = U * n[1, 1 // 2] * n[1, -1 // 2]
# star reference in sector (n_v+1, n_v+1)
Hs, fss, ns = star_hamiltonian(Δ, U, ϵ_imp)
Eex = lowest_energy(Hs, sector_basis(ns, n_v + 1, n_v + 1))
println("exact star sector energy: ", Eex)
for (a, b) in ((1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1))
    (a <= n_v && b <= n_c) || continue
    Hn = natural_impurity_orbital_operator(H_nat, H_int, ϵ_imp, fs, a, b)
    E = lowest_energy(Hn, sector_basis(nsites, n_v + 1, n_v + 1))
    @printf("full Operator ordering (n_v_bit=%d, n_c_bit=%d): E0 - exact = %.2e\n", a, b, E - Eex)
end
for (a, b, p) in ((1, 2, 2), (2, 1, 2), (1, 2, 1), (2, 1, 1))
    (a < n_v && b < n_c) || continue
    fsb = FockSpace(Orbitals(2 + a + b), FermionicSpin(1 // 2))
    nb = occupations(fsb)
    Hb_int = U * nb[1, 1 // 2] * nb[1, -1 // 2]
    H, E0, ψ0 = with_logger(NullLogger()) do
        init_system(Δ, Hb_int, ϵ_imp, ϵ_mf, a, b, p, 1.0e-12)
    end
    @printf("RAS L_v=%d L_c=%d p=%d: E0 - exact = %.2e\n", a, b, p, E0 - Eex)
end

# analytic Bethe Green's function branch vs dense pole sum
G = greens_function_bethe_grid(range(-1, 1; length = 4001))
for z in (-0.5 + 0.1im, 0.5 + 0.1im, -1.5 + 0.1im, 1.5 + 0.1im, 0.0 + 0.1im, -0.3 + 0.001im)
    ga = greens_function_bethe_analytic(z)
    gp = evaluate(G, z)
    @printf("Bethe analytic z=%s: G=%.5f%+.5fi  poles: %.5f%+.5fi  Im<0: %s\n", string(z), real(ga), imag(ga), real(gp), imag(gp), imag(ga) < 0)
end
# Gaussian broadening vs Lorentzian-limit consistency: compare evaluate_gaussian with numerical KK
P = PolesSum([0.3], [1.0])
σ = 0.2
ω = 0.55
gg = evaluate_gaussian(P, ω, σ)
# numerical Hilbert transform of Gaussian spectral function
ws = range(-8, 8; length = 200001)
A = exp.(-(ws .- 0.3) .^ 2 ./ (2σ^2)) ./ (σ * sqrt(2π))
re = sum(A[i] / (ω - ws[i]) for i in eachindex(ws) if abs(ω - ws[i]) > 1e-9) * step(ws)
@printf("Gaussian broadening: code Re=%.6f Im=%.6f | numeric Re=%.6f -πA=%.6f\n", real(gg), imag(gg), re, -π * exp(-(ω - 0.3)^2 / (2σ^2)) / (σ * sqrt(2π)))
