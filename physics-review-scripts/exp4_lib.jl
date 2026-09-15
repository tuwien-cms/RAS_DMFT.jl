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

lowest_energy(H, dets) = eigmin(sector_matrix(H, dets))

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
