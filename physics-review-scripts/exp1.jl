# Experiment 1: sector selection, start-state spin, ϵ_mf independence in full space.
using RAS_DMFT
using Fermions
using Fermions.Wavefunctions
using LinearAlgebra
using Printf

const K = UInt64

function full_matrix(H::Operator, nbits::Int)
    dim = 2^nbits
    M = zeros(dim, dim)
    for k in 0:(dim - 1)
        ψ = Wavefunction(Dict{K, Float64}(K(k) => 1.0))
        ϕ = H * ψ
        for (det, val) in pairs(ϕ)
            M[Int(det) + 1, k + 1] += val
        end
    end
    return M
end

function sector_indices(nsites::Int)
    # returns Dict (N_dn, N_up) => indices
    d = Dict{Tuple{Int, Int}, Vector{Int}}()
    mask = (K(1) << nsites) - 1
    for k in 0:(2^(2 * nsites) - 1)
        n1 = count_ones(K(k) & mask)
        n2 = count_ones((K(k) >> nsites) & mask)
        push!(get!(d, (n1, n2), Int[]), k + 1)
    end
    return d
end

function spin_ops(fs)
    c = annihilators(fs)
    n = occupations(fs)
    norb = size(c, 1)
    Sp = sum(c[i, 1 // 2]' * c[i, -1 // 2] for i in 1:norb)
    Sz = sum(0.5 * (n[i, 1 // 2] - n[i, -1 // 2]) for i in 1:norb)
    S2 = Sz * Sz + 0.5 * (Sp * Sp' + Sp' * Sp)
    Ntot = sum(n[i, σ] for i in 1:norb, σ in (-1 // 2, 1 // 2))
    return S2, Sz, Ntot
end

function run(Δ, U, ϵ_imp, ϵ_mf)
    H_nat = natural_impurity_orbital(Δ, ϵ_mf)
    n_v, n_c = n_valence(H_nat), n_conduction(H_nat)
    nsites = 2 + n_v + n_c
    fs = FockSpace(Orbitals(nsites), FermionicSpin(1 // 2))
    n = occupations(fs)
    H_int = U * n[1, 1 // 2] * n[1, -1 // 2]
    H = natural_impurity_orbital_operator(H_nat, H_int, ϵ_imp, fs, n_v, n_c)
    M = full_matrix(H, 2 * nsites)
    @assert issymmetric(M) || norm(M - M') < 1.0e-12
    M = Symmetric(M)
    secs = sector_indices(nsites)
    # ground state per sector
    Es = Dict{Tuple{Int, Int}, Float64}()
    for (s, idx) in secs
        Es[s] = eigmin(Symmetric(M[idx, idx]))
    end
    Eglob, sglob = findmin(Es)
    # sector chosen by the code
    N_code = n_v + 1 # per spin
    E_code = Es[(N_code, N_code)]
    # start state
    ψs = Wavefunction_singlet(Dict{K, Float64}, n_v, n_c, 0, 0)
    S2, Sz, Nt = spin_ops(fs)
    s2 = dot(ψs, S2, ψs)
    nt = dot(ψs, Nt, ψs)
    # lowest eigenstate with overlap on the start state
    idx = secs[(N_code, N_code)]
    F = eigen(Symmetric(M[idx, idx]))
    v = zeros(length(idx))
    for (det, val) in pairs(ψs)
        v[findfirst(==(Int(det) + 1), idx)] = val
    end
    ov = abs.(F.vectors' * v)
    i_first = findfirst(>(1.0e-8), ov)
    E_reach = F.values[i_first]
    # spin of the state reached
    ψ_gs = Wavefunction(Dict{K, Float64}(K(idx[j] - 1) => F.vectors[j, i_first] for j in eachindex(idx) if abs(F.vectors[j, i_first]) > 0))
    s2_gs = dot(ψ_gs, S2, ψ_gs)
    n_imp = dot(ψ_gs, n[1, 1 // 2] + n[1, -1 // 2], ψ_gs)
    @printf(
        "U=%.2f ϵ_imp=%+.3f ϵ_mf=%+.3f | n_v=%d n_c=%d | code sector N=%d E=%.6f | global sector %s E=%.6f | start S²=%.3f N=%.1f | reached E=%.6f (gap to sector GS %.2e) S²=%.3f n_imp=%.4f\n",
        U, ϵ_imp, ϵ_mf, n_v, n_c, 2 * N_code, E_code, string(sglob), Eglob, s2, nt,
        E_reach, E_reach - E_code, s2_gs, n_imp,
    )
    return M, Es
end

Δ = PolesSum([-2.0, -1.0, 1.0, 2.0], [0.09, 0.04, 0.16, 0.09])
U = 2.0
println("--- ϵ_mf independence of the full spectrum (same sector) ---")
M0, _ = run(Δ, U, -1.0, 0.0)
M1, _ = run(Δ, U, -1.0, 0.3)
println("max |ΔE| between full spectra ϵ_mf=0 vs 0.3: ", maximum(abs, eigvals(M0) - eigvals(M1)))

println("--- sector scan ---")
for ϵ_imp in (-1.0, -0.5, -0.2, 0.3, -1.8), ϵ_mf in (ϵ_imp, ϵ_imp + U / 2, ϵ_imp + U)
    run(Δ, U, ϵ_imp, ϵ_mf)
end
