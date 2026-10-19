using RAS_DMFT.Combinatorics
using Test

@testset "Combinatorics" begin
    @testset "bit component" begin
        # 0 excitation
        @test ndet0_bit(4, 2, 2) == 6 * 6
        @test ndet0_bit(4, 1, 3) == 4 * 4
        @test ndet0_bit(4, 3, 1) == ndet0_bit(4, 1, 3)

        # 1 excitation
        @test ndet1_bit(4, 2, 2) == 4 * 24
        @test ndet1_bit(4, 1, 2) == 6 + 36 + 16 + 16
        @test ndet1_bit(4, 2, 1) == ndet1_bit(4, 1, 2)

        # 2 excitations
        @test ndet2_bit(4, 2, 2) == 4 * 6 + 4 * 16
        @test ndet2_bit(4, 1, 2) == 24 * 3 + 4 * 4
        @test ndet2_bit(4, 2, 1) == ndet2_bit(4, 1, 2)

        # cumulative sum
        @test ndet_bit(4, 2, 2, 0) == 36
        @test ndet_bit(4, 2, 2, 1) == 36 + 96
        @test ndet_bit(4, 2, 2, 2) == 36 + 96 + 88
        @test_throws DomainError ndet_bit(4, 2, 2, 3)
    end # bit component

    @testset "wave function" begin
        @test ndet0(4, 2, 2, 10, 11) == 36
        @test ndet1(4, 2, 2, 10, 11) == 24 * 2 * (10 + 11)
        @test ndet2(4, 1, 1, 2, 5) == 641 # by brute-force enumeration
        @test ndet(4, 2, 2, 2, 10, 11) == 36 + 1008 + 16176
        @test_throws DomainError ndet(4, 2, 2, 3, 10, 11)
    end # wave function
end # Combinatorics
