using RAS_DMFT
using Fermions.Wavefunctions
using LinearAlgebra
using Test

@testset "orthogonalization" begin
    @testset "_orthonormalize_SVD" begin
        # Matrix{ComplexF64}
        Q = rand(ComplexF64, 10, 4)
        @inferred RAS_DMFT._orthonormalize_SVD(Q)
        Q_new, S_sqrt = RAS_DMFT._orthonormalize_SVD(Q)
        @test norm(Q_new' * Q_new - I) < 50 * eps() # Q_new^† Q_new = 𝟙
        @test norm(Q - Q_new * S_sqrt) < 500 * eps() # Q = Q_new * S^{1/2}
        @test ishermitian(S_sqrt)

        # ill-conditioned: κ = 2^12, forming S = M^† M would lose ϵ κ^2 of orthonormality
        δ = 2.0^-12
        Q0 = [1 2; 2 1; 2 -2] / 3
        B0 = ([1 1; 1 1] + δ * [1 -1; -1 1]) / 2
        M = Q0 * B0
        Q, B = RAS_DMFT._orthonormalize_SVD(M)
        @test norm(Q' * Q - I) < 10 * eps()
        @test norm(B - B0) < 10 * eps()

        # rank-deficient: the dropped direction [1, -1] is zero, not completed
        v = [1.0, 2, 3]
        M = [v v]
        Q, B = RAS_DMFT._orthonormalize_SVD(M)
        @test norm(Q - M / sqrt(28)) < 10 * eps()
        @test norm(B - sqrt(7) * ones(2, 2)) < 10 * eps()

        # zero matrix: every direction is dropped
        M = zeros(3, 2)
        Q, B = RAS_DMFT._orthonormalize_SVD(M)
        @test iszero(Q)
        @test iszero(B)

        # RASWavefunction
        q1 = RASWavefunction(
            Dict(zero(UInt8) => rand(5), one(UInt8) => rand(5)), 4, 1, 1, 1,
        )
        v2 = RASWavefunction(Dict(zero(UInt8) => rand(5)), 4, 1, 1, 1)
        Q = [q1 v2]
        @inferred RAS_DMFT._orthonormalize_SVD(Q)
        Q_new, S_sqrt = RAS_DMFT._orthonormalize_SVD(Q)
        # Q_new^† Q_new = 𝟙
        foo = Matrix{Float64}(undef, 2, 2)
        mul!(foo, Q_new', Q_new)
        @test norm(foo - I) < 1000 * eps()
        # Q = Q_new S^{1/2}
        bar = similar(Q)
        mul!(bar, Q_new, S_sqrt)
        for i in eachindex(Q)
            @test norm(bar[i] - Q[i]) < 1000 * eps()
        end
        @test issymmetric(S_sqrt)
    end # _orthonormalize_SVD

    @testset "_orthonormalize_GramSchmidt!" begin
        V1 = [0 0 3; 0 0 0; 100 * eps() 2 1]
        @test RAS_DMFT._orthonormalize_GramSchmidt!(V1) === V1
        @test V1 == [0 0 1; 0 0 0; 0 1 0]
        # applying again keeps it unchanged
        foo = copy(V1)
        @test RAS_DMFT._orthonormalize_GramSchmidt!(V1) == foo

        # no allocations
        V2 = [0 3 1; 0 0 1; 100 * eps() 8 0]
        @test iszero(@allocated(RAS_DMFT._orthonormalize_GramSchmidt!(V2)))

        # last column becomes linearly dependent
        V3 = [1 0 1; 0 1 1; 0 0 0]
        RAS_DMFT._orthonormalize_GramSchmidt!(V3)
        @test V3 == [1 0 0; 0 1 0; 0 0 0]
    end # _orthonormalize_GramSchmidt!

    @testset "_orthogonalize_states!" begin
        Q_new1 = rand(ComplexF64, 8, 4)
        Q_old0 = rand(ComplexF64, 8, 4)
        Q_old1, _ = RAS_DMFT._orthonormalize_SVD(Q_old0)
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
