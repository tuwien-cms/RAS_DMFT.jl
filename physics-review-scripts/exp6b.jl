# Experiment 6b: RAS with L_v ≠ L_c against exact sector energy (7-pole bath from exp2).
using RAS_DMFT, Fermions, Fermions.Wavefunctions, LinearAlgebra, Logging, Printf
include(joinpath(@__DIR__, "exp4_lib.jl"))
Δ = hybridization_function_bethe_grid(range(-1, 1; length = 7))
Δ = PolesSum(locations(Δ) .+ 0.15, weights(Δ) .* [0.6, 0.8, 1.0, 1.2, 1.3, 1.2, 0.9])
U = 2.0; ϵ_imp = -1.0; ϵ_mf = 0.1
H_nat = natural_impurity_orbital(Δ, ϵ_mf)
n_v, n_c = n_valence(H_nat), n_conduction(H_nat)
Hs, fss, ns = star_hamiltonian(Δ, U, ϵ_imp)
Eex = lowest_energy(Hs, sector_basis(ns, n_v + 1, n_v + 1))
println("n_v=$n_v n_c=$n_c exact E=$Eex")
for (a, b, p) in ((1, 1, 2), (1, 2, 2), (2, 1, 2), (1, 2, 1), (2, 1, 1), (1, 3, 2), (3, 1, 2))
    (a < n_v && b < n_c) || continue
    fsb = FockSpace(Orbitals(2 + a + b), FermionicSpin(1 // 2))
    nb = occupations(fsb)
    Hb_int = U * nb[1, 1 // 2] * nb[1, -1 // 2]
    H, E0, ψ0 = with_logger(NullLogger()) do
        init_system(Δ, Hb_int, ϵ_imp, ϵ_mf, a, b, p, 1.0e-12)
    end
    @printf("RAS L_v=%d L_c=%d p=%d: E0 - exact = %.2e\n", a, b, p, E0 - Eex)
end
