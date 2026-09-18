using Fermions
using Fermions.Lanczos
using Fermions.Wavefunctions
using LinearAlgebra
using RAS_DMFT
using Test

@testset "util" begin
    @testset "init_system" begin
        # parameters
        n_bath = 31
        U = 4.0
        μ = U / 2
        L_v = 1
        L_c = 1
        p = 2
        var = eps()

        E0_target = -21.527949990415227 # target ground state energy
        Δ = hybridization_function_bethe_simple(n_bath)
        fs = FockSpace(Orbitals(2 + L_v + L_c), FermionicSpin(1 // 2))
        n = occupations(fs)
        H_int = U * n[1, -1 // 2] * n[1, 1 // 2]
        H, E0, ψ0 = init_system(Δ, H_int, -μ, 0, L_v, L_c, p, var)
        Hψ = H * ψ0
        variance = Hψ ⋅ Hψ
        @test variance < var
        @test E0 ≈ E0_target rtol = 2.0e-13

        # no PHS by setting ϵ_mf ≠ 0
        H_p, E0_p, ψ_p = init_system(Δ, H_int, -μ, 0.5, L_v, L_c, p, var)
        Hψp = H_p * ψ_p
        variance_plus = Hψp ⋅ Hψp
        @test variance_plus < var
        H_m, E0_m, ψ_m = init_system(Δ, H_int, -μ, -0.5, L_v, L_c, p, var)
        Hψm = H_m * ψ_m
        variance_minus = Hψm ⋅ Hψm
        @test variance_minus < var
        @test E0_p ≈ E0_m rtol = 1.0e-13
        # shift destroys PHS
        @test !isapprox(E0_p, E0; rtol = 1.0e-9)
    end # init system

    @testset "Kondo temperature" begin
        @test temperature_kondo(0.3, -0.1, 0.1) == 0.04297872341114842
        @test temperature_kondo(0.2, -0.1, 0.015) == 0.00020610334475146955
    end # Kondo temperature

    @testset "find chemical potential" begin
        # a level at the Fermi level counts half,
        # so a particle-hole symmetric band is half filled
        H_k = [[0.0;;]]
        Σ_stat = Diagonal([0.0])
        Σ_dyn = PolesSumBlock([-1.0, 1.0], [[0.5;;], [0.5;;]])
        Σ_A = arrowhead_matrix(Σ_dyn; tol_weight = 0, thin = true)
        @test RAS_DMFT._filling_mu(H_k, Σ_stat, Σ_A, 0.0) ≈ 0.5 atol = 10 * eps()
        # PHS: n(-μ) + n(+μ) = 1
        μ = 2.0
        @test RAS_DMFT._filling_mu(H_k, Σ_stat, Σ_A, -μ) +
            RAS_DMFT._filling_mu(H_k, Σ_stat, Σ_A, μ) ≈ 1 atol = 10 * eps()

        # A pole whose weight is supported on a single band couples to that band
        # alone, so in such a frame the arrowhead falls apart into the 2x2 blocks
        # [ϵ+σ-μ √w; √w p],
        # whose spectrum and eigenvectors are known in closed form.
        # Rotating the band block by a unitary moves neither the eigenvalues
        # nor the norm of the band part of an eigenvector,
        # so the filling stays exact once the matrices are dense.
        ϵ_k = [[-0.7, 0.2, 1.1], [-0.4, 0.5, 0.9]] # one band matrix per k-point
        σ = [0.15, -0.3, 0.4]
        p = [-1.2, 0.5]     # the first pole carries the weight of two bands
        w = [0.6, 0.9, 0.35] # weight carried by each band
        pole = [1, 1, 2]     # pole each band couples to
        U = Matrix(qr([1.0 2.0 -1.0; 0.5 -1.0 3.0; -2.0 1.0 0.5]).Q)
        H_k = [Hermitian(U * diagm(ϵ) * U') for ϵ in ϵ_k]
        Σ_stat = Hermitian(U * Diagonal(σ) * U')
        Σ_dyn = PolesSumBlock(
            p,
            [
                Hermitian(U * diagm([pole[b] == j ? w[b] : 0.0 for b in eachindex(w)]) * U')
                    for j in eachindex(p)
            ],
        )
        Σ_A = convert(Matrix{Float64}, arrowhead_matrix(Σ_dyn, 0.0; thin = true))

        # Every eigenvalue below the Fermi level contributes the band part of its
        # eigenvector, which for a 2x2 block is (p - λ)^2 / ((p - λ)^2 + w).
        function filling_exact(μ)
            total = sum(ϵ_k) do ϵ
                sum(eachindex(w)) do b
                    d, p_b, w_b = ϵ[b] + σ[b] - μ, p[pole[b]], w[b]
                    r = sqrt((d - p_b)^2 + 4 * w_b)
                    λ = (0.5 * (d + p_b - r), 0.5 * (d + p_b + r))
                    return sum(l -> l < 0 ? (p_b - l)^2 / ((p_b - l)^2 + w_b) : 0.0, λ)
                end
            end
            return total / length(ϵ_k)
        end

        # These μ keep every level clear of the window around the Fermi level,
        # inside which a level counts half and the closed form above breaks down.
        for μ in (-2.0, -0.5, 0.0, 0.75, 1.7, 3.0)
            @test RAS_DMFT._filling_mu(H_k, Σ_stat, Σ_A, μ) ≈
                filling_exact(μ) atol = 8 * eps()
        end

        # Bisection has to recover the chemical potential the filling came from.
        for (μ_target, n_fill) in (
                (-0.5, 0.34208772893164707),
                (0.75, 1.9084225477985441),
                (1.7, 2.7786451479534993),
            )
            @test filling_exact(μ_target) ≈ n_fill atol = 8 * eps()
            μ, n = find_chemical_potential(
                H_k, Σ_stat, Σ_dyn, n_fill;
                μ_min = -4.0, μ_max = 6.0, μ_tol = 1.0e-13, b_max = 200,
            )
            @test μ ≈ μ_target atol = 1.0e-12
            @test n ≈ n_fill atol = 1.0e-12
        end

        # A single 2x2 block turns singular at μ = ϵ + σ - w / p,
        # putting an eigenvalue exactly at the Fermi level.
        # Its null vector carries p^2 / (p^2 + w) on the band,
        # and half of that weight counts towards the filling.
        ϵ_1, σ_1, p_1, w_1 = 0.3, -0.2, 1.5, 0.8
        H_k = [fill(ϵ_1, 1, 1)]
        Σ_stat = Diagonal([σ_1])
        Σ_dyn = PolesSumBlock([p_1], [fill(w_1, 1, 1)])
        Σ_A = convert(Matrix{Float64}, arrowhead_matrix(Σ_dyn, 0.0; thin = true))
        @test RAS_DMFT._filling_mu(H_k, Σ_stat, Σ_A, ϵ_1 + σ_1 - w_1 / p_1) ≈
            p_1^2 / (2 * (p_1^2 + w_1)) atol = 8 * eps()
    end # find chemical potential

    @testset "_issorted_and_unique" begin
        @test RAS_DMFT._issorted_and_unique(1:10)
        @test_throws ArgumentError RAS_DMFT._issorted_and_unique([1, 0]) # not sorted
        @test_throws ArgumentError RAS_DMFT._issorted_and_unique([0, 0, 1]) # not unique
        # duplicate zeros
        @test_throws ArgumentError RAS_DMFT._issorted_and_unique([-0.0, 0.0])
    end # _issorted_and_unique


end # util
