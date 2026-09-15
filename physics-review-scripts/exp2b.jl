# Experiment 2b: sector energies (even and odd) and RAS truncation vs ϵ_mf, 9-pole bath.
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
    I = Int[]; J = Int[]; V = Float64[]
    for (j, d) in enumerate(dets)
        ϕ = H * Wavefunction(Dict{K, Float64}(d => 1.0))
        for (det, val) in pairs(ϕ)
            push!(I, idx[det]); push!(J, j); push!(V, val)
        end
    end
    return sparse(I, J, V, length(dets), length(dets))
end

# Lanczos with full reorthogonalization, lowest eigenpair
function lowest(M::SparseMatrixCSC, nk = 150)
    n = size(M, 1)
    n <= 2000 && (F = eigen(Symmetric(Matrix(M))); return F.values[1], F.vectors[:, 1])
    Q = zeros(n, nk)
    α = zeros(nk); β = zeros(nk - 1)
    q = normalize(ones(n) .+ 0.1 .* randn(n))
    Q[:, 1] = q
    w = M * q
    α[1] = dot(q, w)
    w -= α[1] * q
    k = nk
    for j in 2:nk
        for _ in 1:2
            w -= Q[:, 1:(j - 1)] * (Q[:, 1:(j - 1)]' * w)
        end
        β[j - 1] = norm(w)
        if β[j - 1] < 1e-12
            k = j - 1
            break
        end
        q = w / β[j - 1]
        Q[:, j] = q
        w = M * q - β[j - 1] * Q[:, j - 1]
        α[j] = dot(q, w)
        w -= α[j] * q
    end
    T = SymTridiagonal(α[1:k], β[1:(k - 1)])
    F = eigen(T)
    v = Q[:, 1:k] * F.vectors[:, 1]
    return F.values[1], normalize(v)
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

function exact_sector(H, fs, nsites, n1, n2)
    n = occupations(fs)
    dets = sector_basis(nsites, n1, n2)
    M = sector_matrix(H, dets)
    E0, v = lowest(M)
    ψ = Wavefunction(Dict{K, Float64}(d => v[i] for (i, d) in enumerate(dets) if abs(v[i]) > 1.0e-14))
    nimp = dot(ψ, n[1, 1 // 2] + n[1, -1 // 2], ψ)
    docc = dot(ψ, n[1, 1 // 2] * n[1, -1 // 2], ψ)
    return E0, nimp, docc
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
    return E0, nimp, docc
end

function hartree_sc(Δ, U, ϵ_imp)
    ϵ = ϵ_imp + U / 2
    for _ in 1:300
        A = arrowhead_matrix(Δ); A[1, 1] = ϵ
        F = eigen(Symmetric(A))
        nσ = sum(abs2, F.vectors[1, F.values .< 0])
        ϵ = 0.7 * ϵ + 0.3 * (ϵ_imp + U * nσ)
    end
    return ϵ
end

# 9-pole asymmetric bath
Δ = hybridization_function_bethe_grid(range(-1, 1; length = 9))
Δ = PolesSum(locations(Δ) .+ 0.12, weights(Δ) .* [0.5, 0.7, 0.9, 1.0, 1.2, 1.3, 1.2, 1.0, 0.8])
println("Δ: ", length(Δ), " poles, moment0 = ", moment(Δ, 0), " moment1 = ", moment(Δ, 1))
U = 2.0
for ϵ_imp in (-1.0, -0.4, -1.5)
    println("=== U=$U ϵ_imp=$ϵ_imp ===")
    H, fs, nsites = star_hamiltonian(Δ, U, ϵ_imp)
    Es = Dict{Tuple{Int, Int}, Tuple{Float64, Float64, Float64}}()
    for (n1, n2) in ((4, 4), (4, 5), (5, 5), (5, 6), (6, 6), (3, 4))
        Es[(n1, n2)] = exact_sector(H, fs, nsites, n1, n2)
        @printf("  exact sector (%d,%d) N=%2d: E=%.6f n_imp=%.4f d=%.4f\n", n1, n2, n1 + n2, Es[(n1, n2)]...)
    end
    Eglob = minimum(x -> x[1], values(Es))
    cands = [("bare", ϵ_imp), ("ϵ_imp+U/2", ϵ_imp + U / 2), ("Hartree SC", hartree_sc(Δ, U, ϵ_imp))]
    for (name, ϵ_mf) in cands
        H_nat = natural_impurity_orbital(Δ, ϵ_mf)
        Ns = n_valence(H_nat) + 1
        Eex, nex, dex = Es[(Ns, Ns)]
        @printf("  %-10s ϵ_mf=%+.4f -> sector N=%d (E_sector - E_global = %.4f)\n", name, ϵ_mf, 2Ns, Eex - Eglob)
        for (L, p) in ((1, 1), (1, 2), (2, 1), (2, 2))
            (L < n_valence(H_nat) && L < n_conduction(H_nat)) || continue
            E0, nimp, docc = ras(Δ, U, ϵ_imp, ϵ_mf, L, L, p)
            @printf("      RAS L=%d p=%d: ΔE=%.2e Δn=%+.2e Δd=%+.2e\n", L, p, E0 - Eex, nimp - nex, docc - dex)
        end
    end
end
