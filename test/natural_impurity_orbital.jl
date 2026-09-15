using Fermions
using LinearAlgebra
using RAS_DMFT
using StaticArrays: @SMatrix
using Test

@testset "natural impurity orbitals" begin
    @testset "NaturalImpurityOrbital" begin
        @testset "constructor" begin
            H_ib = @SMatrix [-1.0 0.5; 0.5 1.0]
            i_v, i_c, b_v, b_c = 0.1, 0.2, 0.3, 0.4
            e_v = [-2.0, -1.5]
            t_v = [0.7]
            e_c = [1.5, 2.0]
            t_c = [0.8]
            H_nat = NaturalImpurityOrbital(H_ib, i_v, i_c, b_v, b_c, e_v, t_v, e_c, t_c)

            @test H_nat.H_ib === H_ib
            @test H_nat.i_v === i_v
            @test H_nat.i_c === i_c
            @test H_nat.b_v === b_v
            @test H_nat.b_c === b_c
            @test H_nat.e_v === e_v
            @test H_nat.t_v === t_v
            @test H_nat.e_c === e_c
            @test H_nat.t_c === t_c

            # non-symmetric H_ib
            H_asym = @SMatrix [-1.0 0.5; 0.6 1.0]
            @test_throws ArgumentError NaturalImpurityOrbital(
                H_asym, i_v, i_c, b_v, b_c, e_v, t_v, e_c, t_c
            )

            # zero valence energy
            @test_throws ArgumentError NaturalImpurityOrbital(
                H_ib, i_v, i_c, b_v, b_c, [-2.0, 0.0], t_v, e_c, t_c
            )
            # positive valence energy
            @test_throws ArgumentError NaturalImpurityOrbital(
                H_ib, i_v, i_c, b_v, b_c, [-2.0, 1.5], t_v, e_c, t_c
            )

            # zero conduction energy
            @test_throws ArgumentError NaturalImpurityOrbital(
                H_ib, i_v, i_c, b_v, b_c, e_v, t_v, [0.0, 2.0], t_c
            )
            # negative conduction energy
            @test_throws ArgumentError NaturalImpurityOrbital(
                H_ib, i_v, i_c, b_v, b_c, e_v, t_v, [1.5, -2.0], t_c
            )

            # valence length mismatch
            @test_throws ArgumentError NaturalImpurityOrbital(
                H_ib, i_v, i_c, b_v, b_c, e_v, [0.7, 0.8], e_c, t_c
            )
            @test_throws ArgumentError NaturalImpurityOrbital(
                H_ib, i_v, i_c, b_v, b_c, e_v, Float64[], e_c, t_c
            )

            # conduction length mismatch
            @test_throws ArgumentError NaturalImpurityOrbital(
                H_ib, i_v, i_c, b_v, b_c, e_v, t_v, e_c, [0.7, 0.8]
            )
            @test_throws ArgumentError NaturalImpurityOrbital(
                H_ib, i_v, i_c, b_v, b_c, e_v, t_v, e_c, Float64[]
            )

            # empty chains
            NaturalImpurityOrbital(
                H_ib, i_v, i_c, b_v, b_c, Float64[], Float64[], Float64[], Float64[]
            )
        end # constructor

        @testset "custom functions" begin
            H_ib = @SMatrix [-1.0 0.5; 0.5 1.0]
            i_v, i_c, b_v, b_c = 0.1, 0.2, 0.3, 0.4
            e_v = [-2.0, -1.5]
            t_v = [0.7]
            e_c = [1.5, 2.0, 1.0]
            t_c = [0.8, 0.9]
            H_nat = NaturalImpurityOrbital(H_ib, i_v, i_c, b_v, b_c, e_v, t_v, e_c, t_c)

            @test n_conduction(H_nat) == 3

            @test n_valence(H_nat) == 2
        end # custom functions

        @testset "Base" begin
            H_ib = @SMatrix [-1.0 0.5; 0.5 1.0]
            i_v, i_c, b_v, b_c = 0.1, 0.2, 0.3, 0.4
            e_v = [-2.0, -1.5]
            t_v = [0.7]
            e_c = [1.5, 2.0, 1.0]
            t_c = [0.8, 0.9]
            H_nat = NaturalImpurityOrbital(H_ib, i_v, i_c, b_v, b_c, e_v, t_v, e_c, t_c)

            @test eltype(H_nat) == Float64
            @test eltype(typeof(H_nat)) == Float64

            H_ref = [
                -1.0  0.5   0.1   0.0  0.2  0.0  0.0
                0.5  1.0   0.3   0.0  0.4  0.0  0.0
                0.1  0.3  -2.0   0.7  0.0  0.0  0.0
                0.0  0.0   0.7  -1.5  0.0  0.0  0.0
                0.2  0.4   0.0   0.0  1.5  0.8  0.0
                0.0  0.0   0.0   0.0  0.8  2.0  0.9
                0.0  0.0   0.0   0.0  0.0  0.9  1.0
            ]
            @test Matrix(H_nat) == H_ref

            @test repr(H_nat) == "7×7 NaturalImpurityOrbital{Float64}"
            @test repr(MIME"text/plain"(), H_nat) ==
                "7×7 NaturalImpurityOrbital{Float64}:\n" *
                sprint(Base.print_matrix, H_ref)

            @test size(H_nat) == (7, 7)
            @test_throws BoundsError size(H_nat, 0)
            @test size(H_nat, 1) == 7
            @test size(H_nat, 2) == 7
            @test size(H_nat, 3) == 1
        end # Base
    end # NaturalImpurityOrbital

    @testset "transformation" begin
        @testset "PHS metal" begin
            Δ = PolesSum([-2.0, -1.0, 0.0, 1.0, 2.0], [0.09, 0.09, 0.1, 0.09, 0.09])
            H_nat = natural_impurity_orbital(Δ)
            M = Matrix(H_nat)
            E = eigvals(Symmetric(arrowhead_matrix(Δ)))
            E_ib = eigvals(Symmetric(H_nat.H_ib))
            E_occ = view(E, 1:searchsortedlast(E, 0))
            E_nat = eigvals(Symmetric(M))

            # same eigenvalues
            @test E_nat ≈ E atol = 1.0e-13
            # i, b at Fermi energy
            @test H_nat.H_ib[1, 1] ≈ 0 atol = 1.0e-13
            @test H_nat.H_ib[2, 2] ≈ 0 atol = 1.0e-13
            # i, b hold one occupied and one empty level
            @test E_ib[1] < 0 < E_ib[2]
            # energy of the occupied levels is basis independent
            @test sum(H_nat.e_v) + E_ib[1] ≈ sum(E_occ) atol = 1.0e-13
            # PHS makes both chains mirror images of each other
            @test H_nat.e_v ≈ -H_nat.e_c atol = 1.0e-13
            @test H_nat.t_v ≈ H_nat.t_c atol = 1.0e-13
            # every bath state hybridizes, sign is chosen to be positive
            @test all(>(0), H_nat.t_v)
            @test all(>(0), H_nat.t_c)
            # impurity keeps its total hybridization, now split over b, v_1 and c_1
            @test H_nat.H_ib[1, 2]^2 + H_nat.i_v^2 + H_nat.i_c^2 ≈
                moment(Δ, 0) atol = 1.0e-13
            # no net hopping: v_1 → i → c_1 cancels with v_1 → b → c_1
            @test H_nat.i_v * H_nat.i_c ≈ -H_nat.b_v * H_nat.b_c atol = 1.0e-14
            # impurity coupling splits evenly in PHS
            @test abs(H_nat.b_v) ≈ abs(H_nat.i_v) atol = 1.0e-13
            @test abs(H_nat.i_c) ≈ abs(H_nat.i_v) atol = 1.0e-13
            @test abs(H_nat.b_c) ≈ abs(H_nat.i_v) atol = 1.0e-13
        end # PHS metal

        @testset "PHS insulator" begin
            Δ = PolesSum([-2.0, -1.0, 1.0, 2.0], fill(0.09, 4))
            H_nat = natural_impurity_orbital(Δ)
            M = Matrix(H_nat)
            E = eigvals(Symmetric(arrowhead_matrix(Δ)))
            E_ib = eigvals(Symmetric(H_nat.H_ib))
            E_occ = view(E, 1:searchsortedlast(E, 0))
            E_nat = eigvals(Symmetric(M))

            # shared Fermi energy
            mid = length(E_nat) ÷ 2
            @test E_nat[mid] ≈ 0 atol = 1.0e-13
            @test E_nat[mid + 1] ≈ 0 atol = 1.0e-13
            deleteat!(E_nat, mid)
            @test E_nat ≈ E atol = 1.0e-13
            # i, b at Fermi energy
            @test H_nat.H_ib[1, 1] ≈ 0 atol = 1.0e-13
            @test H_nat.H_ib[2, 2] ≈ 0 atol = 1.0e-13
            # i, b hold one occupied and one empty level
            @test E_ib[1] < 0 < E_ib[2]
            # energy of the occupied levels is basis independent
            @test sum(H_nat.e_v) + E_ib[1] ≈ sum(E_occ) atol = 1.0e-13
            # PHS makes both chains mirror images of each other
            @test H_nat.e_v ≈ -H_nat.e_c atol = 1.0e-13
            @test H_nat.t_v ≈ H_nat.t_c atol = 1.0e-13
            # every bath state hybridizes, sign is chosen to be positive
            @test all(>(0), H_nat.t_v)
            @test all(>(0), H_nat.t_c)
            # impurity keeps its total hybridization, now split over b, v_1 and c_1
            @test H_nat.H_ib[1, 2]^2 + H_nat.i_v^2 + H_nat.i_c^2 ≈
                moment(Δ, 0) atol = 1.0e-13
            # no net hopping: v_1 → i → c_1 cancels with v_1 → b → c_1
            @test H_nat.i_v * H_nat.i_c ≈ -H_nat.b_v * H_nat.b_c atol = 1.0e-14
            # impurity coupling splits evenly in PHS
            @test abs(H_nat.b_v) ≈ abs(H_nat.i_v) atol = 1.0e-13
            @test abs(H_nat.i_c) ≈ abs(H_nat.i_v) atol = 1.0e-13
            @test abs(H_nat.b_c) ≈ abs(H_nat.i_v) atol = 1.0e-13
        end # PHS insulator

        @testset "no PHS" begin
            Δ = PolesSum([-2.0, -1.0, 1.0, 2.0], [0.09, 0.04, 0.16, 0.09])
            H_nat = natural_impurity_orbital(Δ)
            M = Matrix(H_nat)
            E = eigvals(Symmetric(arrowhead_matrix(Δ)))
            E_ib = eigvals(Symmetric(H_nat.H_ib))
            E_occ = view(E, 1:searchsortedlast(E, 0))
            E_nat = eigvals(Symmetric(M))

            # same eigenvalues
            @test E_nat ≈ E atol = 1.0e-13
            # i at Fermi energy, b is free to sit anywhere without PHS
            @test H_nat.H_ib[1, 1] ≈ 0 atol = 1.0e-13
            # i, b hold one occupied and one empty level
            @test E_ib[1] < 0 < E_ib[2]
            # energy of the occupied levels is basis independent
            @test sum(H_nat.e_v) + E_ib[1] ≈ sum(E_occ) atol = 1.0e-13
            # every bath state hybridizes, sign is chosen to be positive
            @test all(>(0), H_nat.t_v)
            @test all(>(0), H_nat.t_c)
            # impurity keeps its total hybridization, now split over b, v_1 and c_1
            @test H_nat.H_ib[1, 2]^2 + H_nat.i_v^2 + H_nat.i_c^2 ≈
                moment(Δ, 0) atol = 1.0e-13
            # no net hopping: v_1 → i → c_1 cancels with v_1 → b → c_1
            @test H_nat.i_v * H_nat.i_c ≈ -H_nat.b_v * H_nat.b_c atol = 1.0e-14
        end # no PHS

        @testset "Bethe 301" begin
            # Diagonalizing 302 sites is less accurate than the small systems
            # above, so the symmetries only hold to a looser tolerance.
            Δ = hybridization_function_bethe_simple(301)
            H_nat = natural_impurity_orbital(Δ)
            M = Matrix(H_nat)
            E = eigvals(Symmetric(arrowhead_matrix(Δ)))
            E_ib = eigvals(Symmetric(H_nat.H_ib))
            E_occ = view(E, 1:searchsortedlast(E, 0))
            E_nat = eigvals(Symmetric(M))

            # same eigenvalues
            @test E_nat ≈ E atol = 1.0e-13
            # i, b at Fermi energy
            @test H_nat.H_ib[1, 1] ≈ 0 atol = 1.0e-13
            @test H_nat.H_ib[2, 2] ≈ 0 atol = 1.0e-11
            # i, b hold one occupied and one empty level
            @test E_ib[1] < 0 < E_ib[2]
            # energy of the occupied levels is basis independent
            @test sum(H_nat.e_v) + E_ib[1] ≈ sum(E_occ) atol = 1.0e-13
            # PHS makes both chains mirror images of each other
            @test H_nat.e_v ≈ -H_nat.e_c atol = 1.0e-11
            @test H_nat.t_v ≈ H_nat.t_c atol = 1.0e-11
            # every bath state hybridizes, sign is chosen to be positive
            @test all(>(0), H_nat.t_v)
            @test all(>(0), H_nat.t_c)
            # impurity keeps its total hybridization, now split over b, v_1 and c_1
            @test H_nat.H_ib[1, 2]^2 + H_nat.i_v^2 + H_nat.i_c^2 ≈
                moment(Δ, 0) atol = 1.0e-13
            # no net hopping: v_1 → i → c_1 cancels with v_1 → b → c_1
            @test H_nat.i_v * H_nat.i_c ≈ -H_nat.b_v * H_nat.b_c atol = 1.0e-14
            # impurity coupling splits evenly in PHS
            @test abs(H_nat.b_v) ≈ abs(H_nat.i_v) atol = 1.0e-11
            @test abs(H_nat.i_c) ≈ abs(H_nat.i_v) atol = 1.0e-11
            @test abs(H_nat.b_c) ≈ abs(H_nat.i_v) atol = 1.0e-11
        end # Bethe 301



        @testset "regression values" begin
            Δ = PolesSum([-2.0, -1.0, 1.0, 2.0], [0.09, 0.04, 0.16, 0.09])
            H_nat = natural_impurity_orbital(Δ)
            @test H_nat.H_ib[1, 1] ≈ 0 atol = 1.0e-15
            @test H_nat.H_ib[2, 1] ≈ -0.4989095164078891 rtol = 1.0e-12
            @test H_nat.H_ib[2, 2] ≈ 1.1298453001727637 rtol = 1.0e-12
            @test H_nat.i_v ≈ 0.338009682615291 rtol = 1.0e-12
            @test H_nat.b_v ≈ 0.12788916713419668 rtol = 1.0e-12
            @test H_nat.i_c ≈ 0.12976420498718252 rtol = 1.0e-12
            @test H_nat.b_c ≈ -0.3429653873382279 rtol = 1.0e-12
            @test H_nat.e_v ≈ [-1.6970956482922956, -1.2968693527623103] rtol = 1.0e-12
            @test H_nat.t_v ≈ [0.4574613584283322] rtol = 1.0e-12
            @test H_nat.e_c ≈ [1.8641197008818504] rtol = 1.0e-12
            @test isempty(H_nat.t_c)
        end # regression values
    end # transformation

    @testset "operator" begin
        H_nat = NaturalImpurityOrbital(
            (@SMatrix [0.0 1.0; 1.0 2.0]), # H_ib
            3.0,                           # i_v
            4.0,                           # i_c
            5.0,                           # b_v
            6.0,                           # b_c
            [-7.0, -8.0],                  # e_v
            [9.0],                         # t_v
            [10.0, 11.0],                  # e_c
            [12.0],                        # t_c
        )
        ϵ_imp = 13.0
        U = 14.0
        hop(c, σ, i, j, t) = t * (c[i, σ]' * c[j, σ] + c[j, σ]' * c[i, σ])

        @testset "Operator" begin
            fs = FockSpace(Orbitals(6), FermionicSpin(1 // 2))
            c = annihilators(fs)
            n = occupations(fs)
            H_int = U * n[1, 1 // 2] * n[1, -1 // 2]

            # sites [i, b, v_1, c_1, v_2, c_2]
            H_ref = H_int
            for σ in axes(c, 2)
                H_ref += ϵ_imp * n[1, σ]       # i
                H_ref += 2.0 * n[2, σ]         # b
                H_ref += hop(c, σ, 1, 2, 1.0)  # i ↔ b
                H_ref += -7.0 * n[3, σ]        # v_1
                H_ref += hop(c, σ, 1, 3, 3.0)  # i ↔ v_1
                H_ref += hop(c, σ, 2, 3, 5.0)  # b ↔ v_1
                H_ref += 10.0 * n[4, σ]        # c_1
                H_ref += hop(c, σ, 1, 4, 4.0)  # i ↔ c_1
                H_ref += hop(c, σ, 2, 4, 6.0)  # b ↔ c_1
                H_ref += -8.0 * n[5, σ]        # v_2
                H_ref += hop(c, σ, 3, 5, 9.0)  # v_1 ↔ v_2
                H_ref += 11.0 * n[6, σ]        # c_2
                H_ref += hop(c, σ, 4, 6, 12.0) # c_1 ↔ c_2
            end
            @test natural_impurity_orbital_operator(H_nat, H_int, ϵ_imp, fs, 1, 1) == H_ref

            # all bath sites in the bit component reorders them to
            # [i, b, v_1, v_2, c_1, c_2]
            H_ref = H_int
            for σ in axes(c, 2)
                H_ref += ϵ_imp * n[1, σ]       # i
                H_ref += 2.0 * n[2, σ]         # b
                H_ref += hop(c, σ, 1, 2, 1.0)  # i ↔ b
                H_ref += -7.0 * n[3, σ]        # v_1
                H_ref += hop(c, σ, 1, 3, 3.0)  # i ↔ v_1
                H_ref += hop(c, σ, 2, 3, 5.0)  # b ↔ v_1
                H_ref += -8.0 * n[4, σ]        # v_2
                H_ref += hop(c, σ, 3, 4, 9.0)  # v_1 ↔ v_2
                H_ref += 10.0 * n[5, σ]        # c_1
                H_ref += hop(c, σ, 1, 5, 4.0)  # i ↔ c_1
                H_ref += hop(c, σ, 2, 5, 6.0)  # b ↔ c_1
                H_ref += 11.0 * n[6, σ]        # c_2
                H_ref += hop(c, σ, 5, 6, 12.0) # c_1 ↔ c_2
            end
            @test natural_impurity_orbital_operator(H_nat, H_int, ϵ_imp, fs, 2, 2) == H_ref
        end # Operator

        @testset "RASOperator" begin
            fs = FockSpace(Orbitals(4), FermionicSpin(1 // 2))
            c = annihilators(fs)
            n = occupations(fs)
            H_int = U * n[1, 1 // 2] * n[1, -1 // 2]

            # L_v = L_c = 0
            fs0 = FockSpace(Orbitals(2), FermionicSpin(1 // 2))
            c0 = annihilators(fs0)
            n0 = occupations(fs0)
            H_int0 = U * n0[1, 1 // 2] * n0[1, -1 // 2]
            H_ref0 = H_int0
            for σ in axes(c0, 2)
                H_ref0 += ϵ_imp * n0[1, σ]      # i
                H_ref0 += 2.0 * n0[2, σ]        # b
                H_ref0 += hop(c0, σ, 1, 2, 1.0) # i ↔ b
            end
            H0 = natural_impurity_orbital_ras_operator(
                H_nat, H_int0, ϵ_imp, fs0, 0, 0, 2
            )
            @test H0.opbit == H_ref0
            @test H0.nbit == 2
            @test H0.nfilled == 2
            @test H0.nempty == 2


            # L_v = L_c = 1
            H_ref = H_int
            for σ in axes(c, 2)
                H_ref += ϵ_imp * n[1, σ]       # i
                H_ref += 2.0 * n[2, σ]         # b
                H_ref += hop(c, σ, 1, 2, 1.0)  # i ↔ b
                H_ref += -7.0 * n[3, σ]        # v_1
                H_ref += hop(c, σ, 1, 3, 3.0)  # i ↔ v_1
                H_ref += hop(c, σ, 2, 3, 5.0)  # b ↔ v_1
                H_ref += 10.0 * n[4, σ]        # c_1
                H_ref += hop(c, σ, 1, 4, 4.0)  # i ↔ c_1
                H_ref += hop(c, σ, 2, 4, 6.0)  # b ↔ c_1
            end
            for p in 0:2
                H = natural_impurity_orbital_ras_operator(
                    H_nat, H_int, ϵ_imp, fs, 1, 1, p
                )
                @test H.opbit == H_ref
                @test H.nbit == 4
                @test H.nfilled == 1
                @test H.nempty == 1
                @test H.excitation == p
            end

            # a realistic bath still gives a symmetric bit component
            H_bethe = natural_impurity_orbital(hybridization_function_bethe_simple(11))
            fs6 = FockSpace(Orbitals(6), FermionicSpin(1 // 2))
            n6 = occupations(fs6)
            H6 = natural_impurity_orbital_ras_operator(
                H_bethe, U * n6[1, 1 // 2] * n6[1, -1 // 2], ϵ_imp, fs6, 2, 2, 2
            )
            @test issymmetric(H6.opbit)

            # invalid n_v_bit, n_c_bit, excitation
            @test_throws ArgumentError natural_impurity_orbital_ras_operator(
                H_nat, H_int, ϵ_imp, fs, 0, 1, 1
            )
            @test_throws ArgumentError natural_impurity_orbital_ras_operator(
                H_nat, H_int, ϵ_imp, fs, 1, 0, 1
            )
            @test_throws ArgumentError natural_impurity_orbital_ras_operator(
                H_nat, H_int, ϵ_imp, fs, 3, 1, 1
            )
            @test_throws ArgumentError natural_impurity_orbital_ras_operator(
                H_nat, H_int, ϵ_imp, fs, 1, 3, 1
            )
            @test_throws ArgumentError natural_impurity_orbital_ras_operator(
                H_nat, H_int, ϵ_imp, fs, 1, 1, -1
            )
        end # RASOperator
    end # operator
end # natural impurity orbitals
