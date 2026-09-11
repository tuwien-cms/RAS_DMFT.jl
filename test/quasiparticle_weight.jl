using RAS_DMFT
using Test

@testset "quasiparticle weight" begin
    @testset "quasiparticle_weight" begin
        Σ = PolesSum([-0.25, -0.01, 0.5], [1.0, 2.0, 3.0])
        @inferred quasiparticle_weight(Σ)
        @inferred quasiparticle_weight(Σ; λ = 1.0e-2, tol = 1.0)
        @test quasiparticle_weight(Σ) == inv(20029)
        @test quasiparticle_weight(Σ; tol = 1.0) == inv(20029)
        @test quasiparticle_weight(Σ; tol = 1.1) == inv(20013)
        @test quasiparticle_weight(Σ; λ = 1.0e-2) ≈ inv(10028.9696428138) atol = 10 * eps()
        @test quasiparticle_weight(Σ; tol = 1.1, λ = 1.0e-2) ≈ inv(10012.995201919231) atol = 10 * eps()
    end # quasiparticle_weight

    @testset "quasiparticle_weight_inflections" begin
        # single pole with optimum λ = 1
        Σ = PolesSum([1.0], [2.0])
        @inferred quasiparticle_weight_inflections(Σ; λmax = 2.0)
        Σi = PolesSum([1, 2], [3, 4])
        @inferred quasiparticle_weight_inflections(Σi; λmin = 1, λmax = 2) # Vector{Float64}
        @test quasiparticle_weight_inflections(Σ; λmax = 2.0) ≈ [1.0] atol = 2 * eps()
        # outside the search window
        @test isempty(quasiparticle_weight_inflections(Σ; λmax = 0.5))
        @test isempty(quasiparticle_weight_inflections(Σ; λmin = 1.5, λmax = 2.0))
        @test isempty(quasiparticle_weight_inflections(Σ; tol = 3.0))
        # invalid arguments
        @test_throws ArgumentError quasiparticle_weight_inflections(Σ; tol = -1)
        @test_throws ArgumentError quasiparticle_weight_inflections(Σ; λmin = 0)
        @test_throws ArgumentError quasiparticle_weight_inflections(Σ; λmax = 0)

        Σ2 = PolesSum([-1.0, 1.0], [0.5, 0.5])
        @test quasiparticle_weight_inflections(Σ2; λmax = 2.0) ≈ [sqrt(2 / 3)] atol = 2 * eps()

        #  three inflections
        Σ3 = PolesSum([1.0e-4, 1.0], [1.0e-7, 2.0])
        λ3 = quasiparticle_weight_inflections(Σ3; λmax = 2.0)
        @test λ3 ≈ [1.2018504652954801e-4, 1.9681873959735308e-2, 0.9999998333332845] atol = 10 * eps()
    end # quasiparticle_weight_inflections

    @testset "quasiparticle_weight_optimum_regularization" begin
        # single pole: no plateau, returns 0
        Σ = PolesSum([1.0], [2.0])
        @inferred quasiparticle_weight_optimum_regularization(Σ; λmax = 2.0)
        @test quasiparticle_weight_optimum_regularization(Σ; λmax = 2.0) == 0

        # two poles: tiny pole creates a plateau
        Σ3 = PolesSum([1.0e-4, 1.0], [1.0e-7, 2.0])
        @inferred quasiparticle_weight_optimum_regularization(Σ3)
        @test quasiparticle_weight_optimum_regularization(Σ3) ≈ 0.014951703474155864 atol = 10 * eps()
    end # quasiparticle_weight_optimum_regularization

    @testset "_quasiparticle_weight_log_slope" begin
        # log-inflection root at λ^2 = ϵ^2 + w = 3
        Σ = PolesSum([1.0], [2.0])
        @test RAS_DMFT._quasiparticle_weight_log_slope(Σ, 0, sqrt(3) - 10 * eps()) > 0
        @test RAS_DMFT._quasiparticle_weight_log_slope(Σ, 0, sqrt(3)) ≈ 1 / 3 atol = 10 * eps()
        @test RAS_DMFT._quasiparticle_weight_log_slope(Σ, 0, sqrt(3) + 10 * eps()) > 0
    end # _quasiparticle_weight_log_slope

    @testset "_quasiparticle_weight_log_curvature" begin
        # log-inflection root at λ^2 = ϵ^2 + w = 3
        Σ = PolesSum([1.0], [2.0])
        @test RAS_DMFT._quasiparticle_weight_log_curvature(Σ, 0, sqrt(3) - 10 * eps()) > 0
        @test RAS_DMFT._quasiparticle_weight_log_curvature(Σ, 0, sqrt(3)) ≈ 0 atol = 10 * eps()
        @test RAS_DMFT._quasiparticle_weight_log_curvature(Σ, 0, sqrt(3) + 10 * eps()) < 0
    end # _quasiparticle_weight_log_curvature

    @testset "_bisect_sign_change" begin
        # log-inflection root at λ^2 = ϵ^2 + w = 3
        Σ = PolesSum([1.0], [2.0])
        residual = λ -> RAS_DMFT._quasiparticle_weight_log_curvature(Σ, 0, λ)
        λ_root = RAS_DMFT._bisect_sign_change(residual, 1.0, 2.0)
        @test λ_root ≈ sqrt(3.0) atol = 10 * eps()
    end # _bisect_sign_change
end # quasiparticle weight
