using Fermions
using Fermions.Lanczos
using Fermions.Wavefunctions
using LinearAlgebra
using RAS_DMFT
using Random
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
        Random.seed!(0)
        n_b = 4
        n_fill = n_b * 0.7
        H_k = [Hermitian(randn(n_b, n_b)) for _ in 1:10]
        Σ_stat = Diagonal(randn(n_b))
        amps = [randn(4, 2) for _ in 1:20]
        wgts = [Hermitian(amp * amp') for amp in amps]
        Σ_dyn = PolesSumBlock(randn(20) .* 2, wgts)
        μ, n = find_chemical_potential(H_k, Σ_stat, Σ_dyn, n_fill; μ_min = -1, μ_max = 8)
        @test μ ≈ 5.823782742023468 atol = 1.0e-6
        @test n ≈ n_fill atol = 1.0e-7

        # a level at the Fermi level counts half,
        # so a particle-hole symmetric band is half filled
        H_k = [[0.0;;]]
        Σ_stat = Diagonal([0.0])
        Σ_dyn = PolesSumBlock([-1.0, 1.0], [[0.5;;], [0.5;;]])
        Σ_A = arrowhead_matrix(Σ_dyn, 0.0; thin = true)
        @test RAS_DMFT._filling_mu(H_k, Σ_stat, Σ_A, 0.0) ≈ 0.5 atol = 10 * eps()
        # PHS: n(-μ) + n(+μ) = 1
        μ = 2.0
        @test RAS_DMFT._filling_mu(H_k, Σ_stat, Σ_A, -μ) +
            RAS_DMFT._filling_mu(H_k, Σ_stat, Σ_A, μ) ≈ 1 atol = 10 * eps()
    end # find chemical potential

    @testset "_issorted_and_unique" begin
        @test RAS_DMFT._issorted_and_unique(1:10)
        @test_throws ArgumentError RAS_DMFT._issorted_and_unique([1, 0]) # not sorted
        @test_throws ArgumentError RAS_DMFT._issorted_and_unique([0, 0, 1]) # not unique
        # duplicate zeros
        @test_throws ArgumentError RAS_DMFT._issorted_and_unique([-0.0, 0.0])
    end # _issorted_and_unique


end # util
