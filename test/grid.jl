using RAS_DMFT
using Test

@testset "grid" begin
    @testset "logarithmic grid" begin
        @test_throws ArgumentError grid_log(1, 1.0, 10)
        @test_throws ArgumentError grid_log(1, 2.0, 0)
        @test_throws ArgumentError grid_log(-16.0, 2.0, 1)
        @test grid_log(1, 2, 10) == [
            2^(-9), 2^(-8), 2^(-7), 2^(-6), 2^(-5), 2^(-4), 2^(-3), 2^(-2), 2^(-1), 2^(-0),
        ]
        @test grid_log(16.0, 2, 4) == [2.0, 4.0, 8.0, 16.0]
        @test grid_log(100, 500, 1) == [100.0]
    end # logarithmic grid

    @testset "grid interpolate" begin
        @test_throws ArgumentError grid_interpolate([1, -1], 1)
        @test_throws ArgumentError grid_interpolate([-1, -1, 0], 1)
        @test_throws ArgumentError grid_interpolate([-1.0, -0.0, 0.0], 1)
        @test_throws ArgumentError grid_interpolate([-1, 1], -1)
        @test_throws ArgumentError grid_interpolate([-1, 1], 0)
        a = [-5.0, -4.0, -1.0, 0.0, 1.0, 4.0, 5.0]
        @test grid_interpolate(a, 1) == [-5.0, -4.5, -2.5, -0.5, 0.5, 2.5, 4.5, 5.0]
        @test grid_interpolate(a, 2) == [
            -5.0,
            -4.75,
            -4.5,
            -3.5,
            -2.5,
            -1.5,
            -0.5,
            0.0,
            0.5,
            1.5,
            2.5,
            3.5,
            4.5,
            4.75,
            5.0,
        ]
        # integer grid
        @test grid_interpolate([1, 2, 3], 2) == [1.0, 1.25, 1.5, 2.0, 2.5, 2.75, 3.0]
        # single point has no interval
        @test_throws ArgumentError grid_interpolate([1.0], 2)
    end # grid interpolate
end # grid
