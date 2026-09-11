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
