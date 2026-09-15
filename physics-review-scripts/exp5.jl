# Experiment 5: consistency checks of DMFT self-consistency pieces.
using RAS_DMFT
using LinearAlgebra
using Printf
using Random

Random.seed!(1)
# (a) update_hybridization_function: Δ(z) = Δ0(z + μ - Σ_H - Σ(z))
Δ0 = hybridization_function_bethe_simple(21)
Σ = PolesSum(sort(randn(7) .* 1.5), rand(7) .* 0.3)
μ = 0.37; Σ_H = 0.81
Δ = update_hybridization_function(Δ0, μ, Σ_H, Σ)
zs = [0.3 + 0.2im, -1.1 + 0.05im, 2.5 + 1.0im, 0.0 + 0.01im]
for z in zs
    lhs = evaluate(Δ, z)
    rhs = evaluate(Δ0, z + μ - Σ_H - evaluate(Σ, z))
    @printf("(a) z=%s: |Δ_new(z) - Δ0(z+μ-Σ(z))| = %.2e\n", string(z), abs(lhs - rhs))
end
@printf("(a) moments: M0 %.3e (Δ0: %.3e)\n", moment(Δ, 0), moment(Δ0, 0))

# (b) inverse: G = 1/(z - a0 - D)
G = PolesSum(sort(randn(9)), normalize!(rand(9), 1))
a0, D = inverse(G)
for z in zs
    @printf("(b) z=%s: |1/G - (z - a0 - D)| = %.2e\n", string(z), abs(1 / evaluate(G, z) - (z - a0 - evaluate(D, z))))
end

# (c) self_energy_dyson vs direct at half filling with exact relation
# (d) greens_function_local + find_chemical_potential: filling equals n_fill and Tr ∫ A = n
n_b = 3
H_k = [Hermitian(randn(ComplexF64, n_b, n_b)) for _ in 1:8]
Σ_stat = Diagonal(randn(n_b))
amps = [randn(ComplexF64, n_b, 2) for _ in 1:6]
Σ_dyn = PolesSumBlock(sort(randn(6)) .* 2, [Hermitian(a * a') for a in amps])
n_fill = 1.7
μ, n = find_chemical_potential(H_k, Σ_stat, Σ_dyn, n_fill; μ_min = -10, μ_max = 10)
G_loc = greens_function_local(H_k, Σ_stat, Σ_dyn, μ)
@printf("(d) μ=%.5f n(μ)=%.6f target %.2f; from G_loc: filling=%.6f total weight=%.6f (expect %d)\n", μ, n, n_fill, real(tr(filling(G_loc))), real(tr(moment(G_loc, 0))), n_b)
# check G_loc against direct evaluation
z = 0.4 + 0.3im
direct = sum(inv((z + μ) * I - H_k[i] - Σ_stat - evaluate(Σ_dyn, z)) for i in eachindex(H_k)) / length(H_k)
@printf("(d) |G_loc(z) - direct| = %.2e\n", maximum(abs, evaluate(G_loc, z) - direct))

# (e) self_energy_IFG on a synthetic exact block correlator: build C from an exact
# G = 1/(z - ϵ - Δ - Σ) with known Σ and Hartree; check Schur complement recovers Σ_dyn.
ϵ = -0.3; ΣH = 0.9
Σd = PolesSum([-1.5, -0.4, 0.7, 2.1], [0.3, 0.2, 0.25, 0.15])
Δb = PolesSum([-2.0, -1.0, 0.5, 1.5], [0.1, 0.12, 0.08, 0.1])
# G poles via arrowhead of (Δb + Σd) with on-site ϵ+ΣH
P = Δb + Σd
A = arrowhead_matrix(P); A[1, 1] = ϵ + ΣH
F = eigen(Symmetric(A))
Gp = PolesSum(F.values, abs2.(F.vectors[1, :]))
# F̃ = Σd G, Ĩ = Σd + Σd G Σd as pole sums: evaluate numerically on a fine contour instead
zs2 = [0.2 + 0.3im, -0.7 + 0.2im, 1.3 + 0.5im]
# build block C(z) = [[Ĩ, F̃],[F̃, G]] as exact pole sum through the arrowhead trick:
# C is the correlator of (q̃†, d†) for a noninteracting-like model where Σd acts as a bath.
# Represent: G = 1/(z - ϵ - ΣH - Δb - Σd). In the 2-site picture (impurity + "Σ-site"),
# q̃ ∝ coupling to the Σ bath. We can realize it with a single-particle model:
# sites: imp (0), Σ-bath poles, Δ-bath poles; q̃†|0> = Σ_k b_k c_k†|0>.
locs = vcat(locations(Δb), locations(Σd))
amps_ = vcat(sqrt.(weights(Δb)), sqrt.(weights(Σd)))
nS = length(Σd); nD = length(Δb)
Hsp = zeros(1 + nD + nS, 1 + nD + nS)
Hsp[1, 1] = ϵ + ΣH
for k in eachindex(locs)
    Hsp[k + 1, k + 1] = locs[k]; Hsp[1, k + 1] = Hsp[k + 1, 1] = amps_[k]
end
Fsp = eigen(Symmetric(Hsp))
# operator vectors: d† → e_1, q̃† → Σ_k b_k e_{Σ-site k}
vd = zeros(1 + nD + nS); vd[1] = 1
vq = zeros(1 + nD + nS); vq[(2 + nD):end] = sqrt.(weights(Σd))
Vop = [vq vd]
Wts = [Hermitian(Vop' * Fsp.vectors[:, j] * Fsp.vectors[:, j]' * Vop) for j in axes(Fsp.vectors, 2)]
C = PolesSumBlock(Fsp.values, Wts)
Σ_ifg = PolesSum(self_energy_IFG(C, 1), 1, 1)
for z in zs2
    @printf("(e) z=%s: |Σ_IFG(z) - Σ_dyn(z)| = %.2e\n", string(z), abs(evaluate(Σ_ifg, z) - evaluate(Σd, z)))
end
@printf("(e) Σ_IFG poles: %s\n", string(round.(locations(Σ_ifg); digits = 6)))
@printf("(e) Σ_IFG weights: %s\n", string(round.(weights(Σ_ifg); digits = 6)))

# (f) Bethe discretizations: moments of the semicircle (D=1): M0=1, M2=1/4, M4=1/8
for (name, G) in (("simple 51", greens_function_bethe_simple(51)), ("grid 51 on [-1,1]", greens_function_bethe_grid(range(-1, 1; length = 51))), ("equal weight 51", greens_function_bethe_equal_weight(51)), ("grid 201 on [-1,1]", greens_function_bethe_grid(range(-1, 1; length = 201))))
    @printf("(f) %-20s M0-1=%+.2e M1=%+.2e M2-1/4=%+.2e M4-1/8=%+.2e\n", name, moment(G, 0) - 1, moment(G, 1), moment(G, 2) - 0.25, moment(G, 4) - 0.125)
end
