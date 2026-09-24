using RAS_DMFT
using Fermions.Wavefunctions
using LinearAlgebra
using Test

@testset "orthogonalization" begin
    @testset "_orthonormalize_lowdin" begin
        # Matrix{ComplexF64}
        M = rand(ComplexF64, 10, 4)
        @inferred RAS_DMFT._orthonormalize_lowdin(M)
        Q, B = RAS_DMFT._orthonormalize_lowdin(M)
        @test norm(Q' * Q - I) < 50 * eps() # Q^† Q = 𝟙
        @test norm(M - Q * B) < 500 * eps() # M = Q B
        @test ishermitian(B)

        # ill-conditioned: κ = 2^12, forming S = M^† M would lose ϵ κ^2 of orthonormality
        δ = 2.0^-12
        Q0 = [1 2; 2 1; 2 -2] / 3
        B0 = ([1 1; 1 1] + δ * [1 -1; -1 1]) / 2
        M = Q0 * B0
        Q, B = RAS_DMFT._orthonormalize_lowdin(M)
        @test norm(Q' * Q - I) < 10 * eps()
        @test norm(B - B0) < 10 * eps()

        # rank-deficient: the dropped direction [1, -1] is zero, not completed
        v = [1.0, 2, 3]
        M = [v v]
        Q, B = RAS_DMFT._orthonormalize_lowdin(M)
        @test norm(Q - M / sqrt(28)) < 10 * eps()
        @test norm(B - sqrt(7) * ones(2, 2)) < 10 * eps()

        # zero matrix: every direction is dropped
        M = zeros(3, 2)
        Q, B = RAS_DMFT._orthonormalize_lowdin(M)
        @test iszero(Q)
        @test iszero(B)

        # RASWavefunction
        q1 = RASWavefunction(
            Dict(zero(UInt8) => rand(5), one(UInt8) => rand(5)), 4, 1, 1, 1,
        )
        q2 = RASWavefunction(Dict(zero(UInt8) => rand(5)), 4, 1, 1, 1)
        M = [q1 q2]
        @inferred RAS_DMFT._orthonormalize_lowdin(M)
        Q, B = RAS_DMFT._orthonormalize_lowdin(M)
        # Q^† Q = 𝟙
        foo = Matrix{Float64}(undef, 2, 2)
        mul!(foo, Q', Q)
        @test norm(foo - I) < 1000 * eps()
        # M = Q B
        bar = similar(M)
        mul!(bar, Q, B)
        for i in eachindex(M)
            @test norm(bar[i] - M[i]) < 1000 * eps()
        end
        @test issymmetric(B)
    end # _orthonormalize_lowdin

    @testset "_orthogonalize_states!" begin
        Q_new1 = rand(ComplexF64, 8, 4)
        Q_old0 = rand(ComplexF64, 8, 4)
        Q_old1, _ = RAS_DMFT._orthonormalize_lowdin(Q_old0)
        M1 = Matrix{ComplexF64}(undef, 4, 4)
        @test RAS_DMFT._orthogonalize_states!(M1, Q_new1, Q_old1) === Q_new1
        # overlap to previous state
        foo = norm(Q_old1' * Q_new1)
        @test foo < 1000 * eps()
        # orthogonalize again
        RAS_DMFT._orthogonalize_states!(M1, Q_new1, Q_old1)
        bar = norm(Q_old1' * Q_new1)
        @test bar <= foo
        @test bar < 100 * eps()

        # no allocations
        Q_new2 = rand(ComplexF64, 8, 4)
        @test iszero(@allocated(RAS_DMFT._orthogonalize_states!(M1, Q_new2, Q_old1)))
    end # _orthogonalize_states!
end # orthogonalization
