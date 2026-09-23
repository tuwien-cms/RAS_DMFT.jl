using RAS_DMFT
using LinearAlgebra
using Test

@testset "hybridization" begin
    @testset "Bethe lattice" begin
        @testset "analytic" begin
            # test different number types
            @test hybridization_function_bethe_analytic(-2) == -0.1339745962155614
            @test hybridization_function_bethe_analytic(-true) == -1 // 2
            @test hybridization_function_bethe_analytic(-1 // 2) ==
                -0.25 - 0.4330127018922193im
            @test hybridization_function_bethe_analytic(false) == -0.5im
            @test hybridization_function_bethe_analytic(0.1im) == -0.4524937810560445im
            @test hybridization_function_bethe_analytic(0.5) == 0.25 - 0.4330127018922193im
            @test hybridization_function_bethe_analytic(0x01) == 0.5
            @test hybridization_function_bethe_analytic(2.0) == 0.1339745962155614
            # # vary half-bandwidth
            @test hybridization_function_bethe_analytic(-1, 2) ==
                -0.5 - 0.8660254037844386im
            @test hybridization_function_bethe_analytic(0, 10) == -5im
            # Vector{Complex}
            Δ = hybridization_function_bethe_analytic([2.0 + 0.1im, 3.0 + 0.5im])
            @test typeof(Δ) === Vector{ComplexF64}
            @test length(Δ) === 2
            @test Δ[1] == hybridization_function_bethe_analytic(2.0 + 0.1im)
            @test Δ[2] == hybridization_function_bethe_analytic(3.0 + 0.5im)
        end # analytic

        @testset "simple" begin
            # 101 poles
            Δ = hybridization_function_bethe_simple(101)
            @test typeof(Δ) == PolesSum{Float64, Float64}
            @test length(Δ) === 101
            @test moment(Δ, 0) ≈ 0.25 rtol = 10 * eps()
            @test location(Δ, 51) ≈ 0 atol = 10 * eps()
            @test norm(locations(Δ) + reverse(locations(Δ))) < 50 * eps()
            @test norm(weights(Δ) - reverse(weights(Δ))) < 600 * eps()
            # 100 poles
            Δ = hybridization_function_bethe_simple(100)
            @test typeof(Δ) === PolesSum{Float64, Float64}
            @test length(Δ) === 100
            @test moment(Δ, 0) ≈ 0.25 rtol = 10 * eps()
            @test norm(locations(Δ) + reverse(locations(Δ))) < 100 * eps()
            @test norm(weights(Δ) - reverse(weights(Δ))) < 600 * eps()
            # 101 poles, D = 2
            Δ = greens_function_bethe_simple(101, 2)
            @test moment(Δ, 0) ≈ 1.0 rtol = 10 * eps()
        end # simple

        @testset "grid Hubbard III" begin
            grid = range(-5, 5; length = 101)
            # U = 0
            Δ = hybridization_function_bethe_grid_hubbard3(grid)
            Δ0 = hybridization_function_bethe_grid(grid)
            @test typeof(Δ) === PolesSum{Float64, Float64}
            @test length(Δ) === 101
            @test locations(Δ) == grid
            @test locations(Δ) !== grid
            @test norm(amplitudes(Δ) - amplitudes(Δ0)) < 10 * eps()
            # U = 3
            Δ = hybridization_function_bethe_grid_hubbard3(grid, 3)
            @test amplitudes(Δ)[36] ≈ 0.08918761226820784 atol = 10 * eps()
            @test amplitudes(Δ)[51] == 0
            @test amplitudes(Δ)[66] ≈ 0.08918761226820784 atol = 10 * eps()
            @test moment(Δ, 0) ≈ 0.25 atol = 10 * eps()
        end # grid Hubbard III

        @testset "grid" begin
            # 101 poles
            W = range(-1, 1; length = 101)
            Δ = hybridization_function_bethe_grid(W)
            @test typeof(Δ) === PolesSum{Float64, Float64}
            @test length(Δ) === 101
            @test locations(Δ) == W
            @test locations(Δ) !== W
            @test moment(Δ, 0) ≈ 0.25 rtol = 10 * eps()
            @test norm(weights(Δ) - reverse(weights(Δ))) < 10 * eps()
            @test amplitudes(Δ)[51] ≈ 0.056418488187777546 atol = eps()
            # 100 poles
            W = range(-1, 1; length = 100)
            Δ = hybridization_function_bethe_grid(W)
            @test typeof(Δ) === PolesSum{Float64, Float64}
            @test length(Δ) === 100
            @test locations(Δ) == W
            @test locations(Δ) !== W
            @test moment(Δ, 0) ≈ 0.25 rtol = 10 * eps()
            @test norm(weights(Δ) - reverse(weights(Δ))) < 10 * eps()
            @test amplitudes(Δ)[51] ≈ 0.05670125801017559 atol = eps()
            # 101 poles, D = 2
            W = range(-3, 3; length = 101)
            Δ = hybridization_function_bethe_grid(W, 2)
            @test sum(weights(Δ)) ≈ 1 rtol = 10 * eps()
            @test norm(weights(Δ) - reverse(weights(Δ))) < 10 * eps()
            @test all(iszero, view(amplitudes(Δ), 1:17))
            @test all(iszero, view(amplitudes(Δ), 85:101))
            @test amplitudes(Δ)[51] ≈ 0.13819506847065838 atol = eps()
            # non-equidistant grid
            # Test if dense grid in middle has smaller weights.
            W = [-1:0.01:-0.51; -0.5:0.005:0.5; 0.51:0.01:1]
            Δ = hybridization_function_bethe_grid(W)
            w1 = amplitudes(Δ)[50]
            @test all(i -> i < w1, view(amplitudes(Δ), 51:251))
            @test amplitudes(Δ)[151] ≈ 0.028209464484933045 atol = eps()
        end # grid
    end # Bethe lattice

    @testset "hybridization_function_local" begin
        tol = 1.0e-12

        @testset "single k-point" begin
            # one k-point, two bands: Δ(z) = t^2 / (z + μ - e2) exactly
            # dyadic inputs make the pole, its weight, and ϵ_mf exact in floating point
            e1, e2, t, μ = 0.25, 0.5, 0.5, 0.25
            H_ks = [[e1 t; t e2]]
            Σ_stat = zeros(2, 2)
            Σ_dyn = PolesSumBlock([0.0], [zeros(1, 1)])
            grid = [-1.0, 0.0, 1.0]
            ϵ_mf, Δ = hybridization_function_local(H_ks, Σ_stat, Σ_dyn, [1], μ, grid; tol)
            @test size(ϵ_mf) == (1, 1)
            @test ϵ_mf == [e1 - μ;;]
            @test size(Δ) == (1, 1)
            @test locations(Δ) == grid
            @test locations(Δ) !== grid
            # pole at 0.25 with weight 0.25, split 3:1 between 0 and 1
            @test only.(weights(Δ)) ≈ [0.0, 0.25 * 3 / 4, 0.25 / 4] atol = 1.0e-12

            # pole below the grid lands on the lowest point
            _, Δ = hybridization_function_local(
                H_ks, Σ_stat, Σ_dyn, [1], μ, [0.5, 1.0]; tol
            )
            @test only.(weights(Δ)) ≈ [0.25, 0.0] atol = 1.0e-12
            # pole on a grid point keeps its full weight there
            _, Δ = hybridization_function_local(
                H_ks, Σ_stat, Σ_dyn, [1], μ, [0.0, 0.25, 1.0]; tol
            )
            @test only.(weights(Δ)) ≈ [0.0, 0.25, 0.0] atol = 1.0e-12

            # every orbital correlated: Δ vanishes for a single k-point
            Σ_dyn2 = PolesSumBlock([0.0], [zeros(2, 2)])
            ϵ_mf, Δ = hybridization_function_local(
                H_ks, Σ_stat, Σ_dyn2, [1, 2], μ, grid; tol
            )
            @test ϵ_mf == [e1 - μ t; t e2 - μ]
            @test all(w -> isapprox(w, zeros(2, 2); atol = 1.0e-12), weights(Δ))
        end # single k-point

        @testset "self-energy" begin
            # Two k-points and a dynamic self-energy on band 1.
            # The hard-coded values come from the pole representation:
            # `greens_function_local`, `inverse`, and `to_grid`.
            e1, e2, t, μ = 0.2, 0.5, 0.4, 0.2
            H_ks = [[e1 t; t e2], [e1 - 0.3 0.7t; 0.7t e2 + 0.4]]
            Σ_stat = zeros(2, 2)
            Σ_dyn = PolesSumBlock([-0.4, 0.6], [fill(0.3, 1, 1), fill(0.1, 1, 1)])
            grid = -2.0:0.5:2.0
            ϵ_mf, Δ = hybridization_function_local(H_ks, Σ_stat, Σ_dyn, [1], μ, grid; tol)
            @test only(ϵ_mf) ≈ -0.15 atol = 1.0e-14
            @test locations(Δ) == grid
            @test only.(weights(Δ)) ≈ [
                0.0,
                0.0,
                0.004091668015302498,
                0.00097986516104423,
                0.0205963140552069,
                0.10229110434523458,
                0.013741048423209045,
                0.0,
                0.0,
            ] atol = 1.0e-12
            # sum rule: m_0 = ⟨δM_k^2⟩ + ⟨B_k B_k^†⟩ = 0.0225 + 0.1192
            @test sum(only.(weights(Δ))) ≈ 0.1417 atol = 1.0e-12
        end # self-energy

        @testset "basis and ordering" begin
            # Diagonal bands decouple, so the block is diagonal in the correlated
            # orbitals with the hard-coded scalar weights on the diagonal.
            # Each orbital has two k-points at ϵ ± d, so Δ = d^2 / (z - ϵ - Σ_{dyn}(z)).
            # A self-energy pole of weight w at ϵ gives poles at ϵ ± √w of weight d^2 / 2,
            # no self-energy a single pole at ϵ of weight d^2, all dyadic.
            H_ks_diag = [[-0.625, 0.75, 0.25], [-0.125, 1.25, 1.25]]
            H_ks = Diagonal.(H_ks_diag)
            Σ_stat_diag = [0.125, -0.25, 0.5]
            Σ_stat = Diagonal(Σ_stat_diag)
            locs = [-0.5, 1.0]
            wgts = [[0.25, 0.0, 0.0], [0.0, 0.0, 0.0625]]
            μ = 0.25
            grid = [-1.0, 0.0, 1.0, 1.5]
            ϵ_ref = [-0.5, 0.5, 1.0]
            w_ref = [
                [1, 1, 0, 0] / 32, # poles at -1 and 0, both on the grid
                [0, 1, 1, 0] / 32, # pole at 0.5, split 1:1
                [0, 1, 5, 2] / 32, # poles at 0.75 (1:3) and 1.25 (1:1)
            ]

            # correlated orbitals 1 and 3, in either order
            for idx in ([1, 3], [3, 1])
                Σ_dyn = PolesSumBlock(locs, [Diagonal(wgt[idx]) for wgt in wgts])
                ϵ_mf, Δ = hybridization_function_local(
                    H_ks, Σ_stat, Σ_dyn, idx, μ, grid; tol
                )
                @test ϵ_mf == Diagonal(ϵ_ref[idx])
                for (j, w) in enumerate(weights(Δ))
                    @test w ≈ Diagonal([w_ref[i][j] for i in idx]) atol = 1.0e-12
                end
            end

            # everything rotated into a dense complex basis
            M = ComplexF64[1 2im -1; 0.5im -1 3; -2 1 0.5im]
            U = Matrix(qr(M).Q)
            H_rot = [Hermitian(U * Diagonal(H_k) * U') for H_k in H_ks_diag]
            Σ_stat_rot = Hermitian(U * Diagonal(Σ_stat_diag) * U')
            Σ_dyn_rot = PolesSumBlock(
                locs, [Hermitian(U * diagm(wgt) * U') for wgt in wgts]
            )
            ϵ_mf, Δ = hybridization_function_local(
                H_rot, Σ_stat_rot, Σ_dyn_rot, 1:3, μ, grid; tol
            )
            @test ϵ_mf ≈ U * Diagonal(ϵ_ref) * U' atol = 1.0e-14
            for (j, w) in enumerate(weights(Δ))
                @test w ≈ U * Diagonal([w_ref[i][j] for i in 1:3]) * U' atol = 1.0e-12
            end
        end # basis and ordering

        @testset "errors" begin
            H_ks = [Diagonal([0.1, 0.2])]
            Σ_stat = zeros(2, 2)
            Σ_dyn = PolesSumBlock([0.0], [zeros(1, 1)])
            grid = [-1.0, 1.0]
            # Σ_stat larger than H_k
            @test_throws DimensionMismatch hybridization_function_local(
                H_ks, zeros(3, 3), Σ_dyn, [1], 0.0, grid
            )
            # Σ_dyn is 1×1 but idx has two orbitals
            @test_throws DimensionMismatch hybridization_function_local(
                H_ks, Σ_stat, Σ_dyn, [1, 2], 0.0, grid
            )
            # orbital 3 outside the two bands
            @test_throws ArgumentError hybridization_function_local(
                H_ks, Σ_stat, Σ_dyn, [3], 0.0, grid
            )
            # grid not sorted
            @test_throws ArgumentError hybridization_function_local(
                H_ks, Σ_stat, Σ_dyn, [1], 0.0, [1.0, -1.0]
            )
            # tolerance not positive
            @test_throws DomainError hybridization_function_local(
                H_ks, Σ_stat, Σ_dyn, [1], 0.0, grid; tol = 0
            )
        end # errors
    end # hybridization_function_local
end # hybridization
