using Test
using PhysicalFDM

@testset "Diff Matrix" begin
    @testset "1D" begin
        @test diff_mat(3,1) == [0.0 0.5 0.0; -0.5 0.0 0.5; 0.0 -0.5 0.0]
        @test diff_mat(3,2) ≈ [-2.0 1.0 0.0; 1.0 -2.0 1.0; 0.0 1.0 -2.0]
    end

    @testset "2D" begin
        @test diff_mat2_x(3,3,1) == [
            0.0   0.0   0.0   0.5   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.5   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.5  0.0  0.0  0.0
           -0.5   0.0   0.0   0.0   0.0   0.0  0.5  0.0  0.0
            0.0  -0.5   0.0   0.0   0.0   0.0  0.0  0.5  0.0
            0.0   0.0  -0.5   0.0   0.0   0.0  0.0  0.0  0.5
            0.0   0.0   0.0  -0.5   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0  -0.5   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0  -0.5  0.0  0.0  0.0
        ]
        @test diff_mat2_y(3,3,1) == [
            0.0   0.5  0.0   0.0   0.0  0.0   0.0   0.0  0.0
           -0.5   0.0  0.5   0.0   0.0  0.0   0.0   0.0  0.0
            0.0  -0.5  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.5  0.0   0.0   0.0  0.0
            0.0   0.0  0.0  -0.5   0.0  0.5   0.0   0.0  0.0
            0.0   0.0  0.0   0.0  -0.5  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.5  0.0
            0.0   0.0  0.0   0.0   0.0  0.0  -0.5   0.0  0.5
            0.0   0.0  0.0   0.0   0.0  0.0   0.0  -0.5  0.0
        ]
        @test diff_mat2_x(3,3,2) ≈ [
           -2.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0
            0.0  -2.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0
            0.0   0.0  -2.0   0.0   0.0   1.0   0.0   0.0   0.0
            1.0   0.0   0.0  -2.0   0.0   0.0   1.0   0.0   0.0
            0.0   1.0   0.0   0.0  -2.0   0.0   0.0   1.0   0.0
            0.0   0.0   1.0   0.0   0.0  -2.0   0.0   0.0   1.0
            0.0   0.0   0.0   1.0   0.0   0.0  -2.0   0.0   0.0
            0.0   0.0   0.0   0.0   1.0   0.0   0.0  -2.0   0.0
            0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0  -2.0
        ]
        @test diff_mat2_y(3,3,2) ≈ [
           -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            1.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   1.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0  -2.0   1.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   1.0  -2.0   1.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   1.0  -2.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0  -2.0   1.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   1.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0
        ]
    end

    @testset "3D" begin
        @test diff_mat3_x(3,3,3,1) == [
            0.0   0.0   0.0   0.5   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.5   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.5  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0
           -0.5   0.0   0.0   0.0   0.0   0.0  0.5  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0
            0.0  -0.5   0.0   0.0   0.0   0.0  0.0  0.5  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0
            0.0   0.0  -0.5   0.0   0.0   0.0  0.0  0.0  0.5   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0  -0.5   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0  -0.5   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0  -0.5  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.5   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.5   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.5  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  -0.5   0.0   0.0   0.0   0.0   0.0  0.5  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0  -0.5   0.0   0.0   0.0   0.0  0.0  0.5  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0  -0.5   0.0   0.0   0.0  0.0  0.0  0.5   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0  -0.5   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0  -0.5   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0  -0.5  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.5   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.5   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.5  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  -0.5   0.0   0.0   0.0   0.0   0.0  0.5  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0  -0.5   0.0   0.0   0.0   0.0  0.0  0.5  0.0
            0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0  -0.5   0.0   0.0   0.0  0.0  0.0  0.5
            0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0  -0.5   0.0   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0  -0.5   0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0   0.0   0.0   0.0   0.0   0.0  -0.5  0.0  0.0  0.0
        ]
        @test diff_mat3_y(3,3,3,1) == [
            0.0   0.5  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0
           -0.5   0.0  0.5   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0  -0.5  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.5  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0  -0.5   0.0  0.5   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0  -0.5  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.5  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0  -0.5   0.0  0.5   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0  -0.5  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.5  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0  -0.5   0.0  0.5   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0  -0.5  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.5  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0  -0.5   0.0  0.5   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0  -0.5  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.5  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0  -0.5   0.0  0.5   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0  -0.5  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.5  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0  -0.5   0.0  0.5   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0  -0.5  0.0   0.0   0.0  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.5  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0  -0.5   0.0  0.5   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0  -0.5  0.0   0.0   0.0  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.5  0.0
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0  -0.5   0.0  0.5
            0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0   0.0  0.0   0.0  -0.5  0.0
        ]
        @test diff_mat3_z(3,3,3,1) == [
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.5   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.5   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.5   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.5   0.0   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.5   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.5   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.5  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
           -0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.5  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0  -0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.5  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0   0.0  -0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.5  0.0  0.0  0.0  0.0  0.0  0.0
            0.0   0.0   0.0  -0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.5  0.0  0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0  -0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0  0.5  0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0  -0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.5  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0  -0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.5  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0  -0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.5  0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.5
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -0.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -0.5   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -0.5   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -0.5   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -0.5   0.0   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -0.5   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -0.5   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -0.5  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
        ]
        @test diff_mat3_x(3,3,3,2) ≈ [
           -2.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0  -2.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0  -2.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            1.0   0.0   0.0  -2.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   1.0   0.0   0.0  -2.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   1.0   0.0   0.0  -2.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   1.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   1.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0  -2.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0  -2.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0  -2.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   1.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0  -2.0   0.0   0.0   1.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0  -2.0   0.0   0.0   1.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0  -2.0   0.0   0.0   1.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0  -2.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0  -2.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0  -2.0
        ]
        @test diff_mat3_y(3,3,3,2) ≈ [
           -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            1.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   1.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   1.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   1.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   1.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   1.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   1.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   1.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0
        ]
        @test diff_mat3_z(3,3,3,2) ≈ [
           -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0   0.0
            0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  -2.0
        ]
    end
end

d1 = [1,2,1]
d2 = [1 1 1; 1 2 1; 1 1 1]
d3 = ones(3,3,3)
d3[2,2,2] = 2

@testset "Gradience" begin
    # zero padding by default
    @test grad_central(1, d1) == [1, 0, -1]
    @test grad_central(1, 1, d2) == ([0.5 1.0 0.5; 0.0 0.0 0.0; -0.5 -1.0 -0.5], [0.5 0.0 -0.5; 1.0 0.0 -1.0; 0.5 0.0 -0.5])
    @test grad_central(1, 1, 1, d3) == ([0.5 0.5 0.5; 0.0 0.0 0.0; -0.5 -0.5 -0.5;;; 0.5 1.0 0.5; 0.0 0.0 0.0; -0.5 -1.0 -0.5;;; 0.5 0.5 0.5; 0.0 0.0 0.0; -0.5 -0.5 -0.5], [0.5 0.0 -0.5; 0.5 0.0 -0.5; 0.5 0.0 -0.5;;; 0.5 0.0 -0.5; 1.0 0.0 -1.0; 0.5 0.0 -0.5;;; 0.5 0.0 -0.5; 0.5 0.0 -0.5; 0.5 0.0 -0.5], [0.5 0.5 0.5; 0.5 1.0 0.5; 0.5 0.5 0.5;;; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0;;; -0.5 -0.5 -0.5; -0.5 -1.0 -0.5; -0.5 -0.5 -0.5])
end

@testset "Laplace" begin
    @test laplace(d1, 1)
end