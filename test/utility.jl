using Fermions
using Fermions.Lanczos
using Fermions.Wavefunctions
using LinearAlgebra
using RAS_DMFT
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
        @test E0 ≈ E0_target rtol = 2.0e-15

        # no PHS by setting ϵ_mf ≠ 0
        H_p, E0_p, ψ_p = init_system(Δ, H_int, -μ, 0.5, L_v, L_c, p, var)
        Hψp = H_p * ψ_p
        variance_plus = Hψp ⋅ Hψp
        @test variance_plus < var
        H_m, E0_m, ψ_m = init_system(Δ, H_int, -μ, -0.5, L_v, L_c, p, var)
        Hψm = H_m * ψ_m
        variance_minus = Hψm ⋅ Hψm
        @test variance_minus < var
        @test E0_p ≈ E0_m rtol = 2.0e-15
        # shift destroys PHS
        @test !isapprox(E0_p, E0; rtol = 1.0e-9)

        # short Lanczos runs with many restarts reach the same ground state
        H_k, E0_k, ψ_k = init_system(
            Δ, H_int, -μ, 0, L_v, L_c, p, var; n_kryl = 5, n_max_restart = 100
        )
        Hψk = H_k * ψ_k
        @test Hψk ⋅ Hψk < var
        @test E0_k ≈ E0_target rtol = 2.0e-15
        # a single short restart cannot reach the variance
        @test_logs (:warn,) init_system(
            Δ, H_int, -μ, 0, L_v, L_c, p, var; n_kryl = 2, n_max_restart = 1
        )
        # integer ϵ_imp or U fail inside, so the signature rejects them
        rest = (Int, Int, Int, Int, Float64) # ϵ_mf, L_v, L_c, p, var
        @test !hasmethod(init_system, Tuple{typeof(Δ), typeof(H_int), Int, rest...})
        @test !hasmethod(init_system, Tuple{typeof(Δ), typeof(n[1, 1 // 2]), Float64, rest...})
    end # init system

    @testset "Kondo temperature" begin
        @test temperature_kondo(0.3, -0.1, 0.1) == 0.04297872341114842
        @test temperature_kondo(0.2, -0.1, 0.015) == 0.00020610334475146955
    end # Kondo temperature

    # Each band is diagonal, so the bands decouple.
    # Returns the filling per band.
    function fillings_exact(μ, H_ks, Σ_stat, locs, wgts; tol = 100 * eps())
        n_p = length(locs)
        result = zeros(length(Σ_stat))
        for H_k in H_ks, i in eachindex(Σ_stat)
            A = zeros(1 + n_p, 1 + n_p)
            A[1, 1] = H_k[i] + Σ_stat[i] - μ
            for j in 1:n_p
                A[1 + j, 1 + j] = locs[j]
                A[1, 1 + j] = A[1 + j, 1] = sqrt(wgts[j][i])
            end
            F = eigen(Symmetric(A))
            for q in eachindex(F.values)
                λ, wgt_q = F.values[q], abs2(F.vectors[1, q])
                λ < -tol && (result[i] += wgt_q)
                abs(λ) <= tol && (result[i] += 0.5 * wgt_q)
            end
        end
        return result / length(H_ks)
    end

    @testset "find_chemical_potential" begin
        n_tol = 1.0e-12

        @testset "PHS" begin
            H_ks_diag = [[0.0]]
            Σ_stat_diag = [0.0]
            locs = [-1.0, 1.0]
            wgts_diag = [[0.5], [0.5]]
            H_ks = Diagonal.(H_ks_diag)
            Σ_stat = Diagonal(Σ_stat_diag)
            Σ_dyn = PolesSumBlock(locs, [Diagonal(wgt) for wgt in wgts_diag])

            μ1_ref = -0.5
            n1_ref = sum(fillings_exact(μ1_ref, H_ks_diag, Σ_stat_diag, locs, wgts_diag))
            μ1, n1 = find_chemical_potential(
                H_ks, Σ_stat, Σ_dyn, [1], n1_ref;
                μ_min = -1.0, μ_max = 1.0, μ_tol = 1.0e-13, b_max = 200, n_tol,
            )
            @test μ1 ≈ μ1_ref atol = 1.0e-12
            @test n1 ≈ n1_ref atol = 1.0e-12

            μ2_ref = 0.5
            n2_ref = sum(fillings_exact(μ2_ref, H_ks_diag, Σ_stat_diag, locs, wgts_diag))
            μ2, n2 = find_chemical_potential(
                H_ks, Σ_stat, Σ_dyn, [1], n2_ref;
                μ_min = -1.0, μ_max = 1.0, μ_tol = 1.0e-13, b_max = 200, n_tol,
            )
            @test μ2 ≈ μ2_ref atol = 1.0e-12
            @test n2 ≈ n2_ref atol = 1.0e-12

            @test n1 + n2 ≈ 1 atol = 1.0e-12 # PHS
            # filling has a jump at exactly zero
            # approach symmetric
            μ0, _ = find_chemical_potential(
                H_ks, Σ_stat, Σ_dyn, [1], 0.5;
                μ_min = -1.0, μ_max = 1.0, μ_tol = 1.0e-13, b_max = 200, n_tol,
            )
            @test μ0 ≈ 0 atol = 1.0e-12
            # approach from left
            μ0, n0 = find_chemical_potential(
                H_ks, Σ_stat, Σ_dyn, [1], 0.5;
                μ_min = -1.0, μ_max = 0.0, μ_tol = 1.0e-13, b_max = 200, n_tol,
            )
            @test μ0 ≈ 0 atol = 1.0e-12
            @test n0 ≈ 0.25 atol = 1.0e-12
            # approach from right
            μ0, n0 = find_chemical_potential(
                H_ks, Σ_stat, Σ_dyn, [1], 0.5;
                μ_min = 0.0, μ_max = 1.0, μ_tol = 1.0e-13, b_max = 200, n_tol,
            )
            @test μ0 ≈ 0 atol = 1.0e-12
            @test n0 ≈ 0.75 atol = 1.0e-12
            # an unreachable tolerance exhausts maxevals
            @test_logs (:warn, "filling integral missed n_tol") match_mode = :any begin
                find_chemical_potential(
                    H_ks, Σ_stat, Σ_dyn, [1], 0.5;
                    μ_min = -1.0, μ_max = 0.6, b_max = 1, n_tol = 1.0e-300,
                )
            end
        end # PHS

        @testset "basis and ordering" begin
            H_ks_diag = [[-0.7, 0.2, 1.1], [-0.4, 0.5, 0.9]]
            H_ks = Diagonal.(H_ks_diag)
            Σ_stat_diag = [0.15, -0.3, 0.4]
            Σ_stat = Diagonal(Σ_stat_diag)
            locs = [-1.2, 0.5]
            wgts = [[0.6, 0.0, 0.0], [0.0, 0.0, 0.35]]

            # dynamic self-energy only on bands 1 and 3, in either order
            idx13 = [1, 3]
            idx31 = [3, 1]
            Σ_dyn_13 = PolesSumBlock(locs, [Diagonal(wgt[idx13]) for wgt in wgts])
            Σ_dyn_31 = PolesSumBlock(locs, [Diagonal(wgt[idx31]) for wgt in wgts])

            # rotate by unitary matrix
            M = ComplexF64[1 2im -1; 0.5im -1 3; -2 1 0.5im]
            @test rank(M) == 3
            U = Matrix(qr(M).Q)
            H_rot = [Hermitian(U * Diagonal(H_k) * U') for H_k in H_ks_diag]
            Σ_stat_rot = Hermitian(U * Diagonal(Σ_stat_diag) * U')
            Σ_dyn_rot = PolesSumBlock(
                locs, [Hermitian(U * diagm(wgt) * U') for wgt in wgts]
            )

            # `Σ_dyn` has to match `idx`, and `idx` has to stay inside the bands.
            @test_throws DimensionMismatch find_chemical_potential(
                H_ks, Σ_stat, Σ_dyn_13, [1], 0.1
            )
            @test_throws ArgumentError find_chemical_potential(
                H_ks, Σ_stat, Σ_dyn_13, [1, 1], 0.1
            )
            @test_throws ArgumentError find_chemical_potential(
                H_ks, Σ_stat, Σ_dyn_13, [1, 4], 0.1
            )

            # The fillings are closed forms of the diagonal case.
            # The same system in three layouts: the correlated block in either
            # order, and everything rotated into a dense complex basis.
            # All three have to return the chemical potential it came from.
            for (μ_target, n_fill) in (
                    (-0.5, 1 / 2 - 23 / (4 * sqrt(1489)) - 29 / (4 * sqrt(1801))),
                    (0.75, 9 / 4 - 1 / (4 * sqrt(561))),
                    (1.7, 5 / 2 + sqrt(21) / 36 + 9 / (4 * sqrt(221))),
                )
                @test sum(fillings_exact(μ_target, H_ks_diag, Σ_stat_diag, locs, wgts)) ≈
                    n_fill atol = 8 * eps()
                for (H, Σ_s, Σ_d, idx) in (
                        (H_ks, Σ_stat, Σ_dyn_13, idx13),
                        (H_ks, Σ_stat, Σ_dyn_31, idx31),
                        (H_rot, Σ_stat_rot, Σ_dyn_rot, 1:3),
                    )
                    μ, n = find_chemical_potential(
                        H, Σ_s, Σ_d, idx, n_fill;
                        μ_min = -4.0, μ_max = 6.0, μ_tol = 1.0e-13, b_max = 200, n_tol,
                    )
                    @test μ ≈ μ_target atol = 1.0e-12
                    @test n ≈ n_fill atol = 1.0e-12
                end
            end

            # default bracket is the band range of H_k + Σ_stat, [-0.55, 1.5]
            μ, _ = find_chemical_potential(
                H_ks, Σ_stat, Σ_dyn_13, idx13, 9 / 4 - 1 / (4 * sqrt(561));
                μ_tol = 1.0e-13, b_max = 200, n_tol,
            )
            @test μ ≈ 0.75 atol = 1.0e-12
            # μ = 1.7 lies above the bands
            @test_throws ArgumentError find_chemical_potential(
                H_ks, Σ_stat, Σ_dyn_13, idx13, 5 / 2 + sqrt(21) / 36 + 9 / (4 * sqrt(221))
            )
        end # basis and ordering
    end # find_chemical_potential

    @testset "density_matrix_local" begin
        n_tol = 1.0e-12

        @testset "PHS" begin
            H_ks = [Diagonal([0.0])]
            Σ_stat = Diagonal([0.0])
            Σ_dyn = PolesSumBlock([-1.0, 1.0], [Diagonal([0.5]), Diagonal([0.5])])

            for μ in (-0.5, 0.5)
                ρ_ref = only(fillings_exact(μ, [[0.0]], [0.0], [-1.0, 1.0], [[0.5], [0.5]]))
                ρ = @inferred density_matrix_local(H_ks, Σ_stat, Σ_dyn, [1], μ; n_tol)
                @test ρ isa Hermitian{Float64, Matrix{Float64}}
                @test only(ρ) ≈ ρ_ref atol = 4 * eps()
            end
            # G(z) = (z^2 - 1) / (z (z^2 - 2)) has weight 1/4 at ±√2, and 1/2 at 0
            ρ = density_matrix_local(H_ks, Σ_stat, Σ_dyn, [1], 0.0; n_tol)
            @test only(ρ) ≈ 1 / 4 + 1 / 2 * 1 / 2 atol = 4 * eps()

            # an unreachable tolerance exhausts maxevals
            msg = "density matrix integral missed n_tol"
            @test_logs (:warn, msg) match_mode = :any begin
                density_matrix_local(H_ks, Σ_stat, Σ_dyn, [1], 0.6; n_tol = 1.0e-300)
            end
        end # PHS

        @testset "basis and ordering" begin
            # the diagonal model of `find_chemical_potential`
            H_ks_diag = [[-0.7, 0.2, 1.1], [-0.4, 0.5, 0.9]]
            H_ks = Diagonal.(H_ks_diag)
            Σ_stat_diag = [0.15, -0.3, 0.4]
            Σ_stat = Diagonal(Σ_stat_diag)
            locs = [-1.2, 0.5]
            wgts = [[0.6, 0.0, 0.0], [0.0, 0.0, 0.35]]
            idx13 = [1, 3]
            idx31 = [3, 1]
            Σ_dyn_13 = PolesSumBlock(locs, [Diagonal(wgt[idx13]) for wgt in wgts])
            Σ_dyn_31 = PolesSumBlock(locs, [Diagonal(wgt[idx31]) for wgt in wgts])

            # rotate by unitary matrix
            M = ComplexF64[1 2im -1; 0.5im -1 3; -2 1 0.5im]
            U = Matrix(qr(M).Q)
            H_rot = [Hermitian(U * Diagonal(H_k) * U') for H_k in H_ks_diag]
            Σ_stat_rot = Hermitian(U * Diagonal(Σ_stat_diag) * U')
            Σ_dyn_rot = PolesSumBlock(
                locs, [Hermitian(U * diagm(wgt) * U') for wgt in wgts]
            )

            # `Σ_dyn` has to match `idx`, and `idx` and `orbitals` stay inside the bands.
            @test_throws DimensionMismatch density_matrix_local(
                H_ks, Σ_stat, Σ_dyn_13, [1], 0.1
            )
            @test_throws ArgumentError density_matrix_local(
                H_ks, Σ_stat, Σ_dyn_13, [1, 1], 0.1
            )
            @test_throws ArgumentError density_matrix_local(
                H_ks, Σ_stat, Σ_dyn_13, idx13, 0.1; orbitals = [2, 2]
            )
            @test_throws ArgumentError density_matrix_local(
                H_ks, Σ_stat, Σ_dyn_13, idx13, 0.1; orbitals = [4]
            )
            @test_throws DomainError density_matrix_local(
                H_ks, Σ_stat, Σ_dyn_13, idx13, 0.1; n_tol = 0.0
            )

            # below, inside, and above the bands
            for μ in (-0.5, 0.75, 1.7)
                ρ_ref = Diagonal(fillings_exact(μ, H_ks_diag, Σ_stat_diag, locs, wgts))
                for (Σ_d, idx) in ((Σ_dyn_13, idx13), (Σ_dyn_31, idx31))
                    ρ = @inferred density_matrix_local(H_ks, Σ_stat, Σ_d, idx, μ; n_tol)
                    @test ρ isa Hermitian{Float64, Matrix{Float64}}
                    @test ρ ≈ ρ_ref atol = 4 * eps()
                    ρ_31 = density_matrix_local(
                        H_ks, Σ_stat, Σ_d, idx, μ; orbitals = [3, 1], n_tol
                    )
                    @test ρ_31 ≈ ρ_ref[[3, 1], [3, 1]] atol = 4 * eps()
                end

                ρ_rot_ref = U * ρ_ref * U'
                ρ_rot = @inferred density_matrix_local(
                    H_rot, Σ_stat_rot, Σ_dyn_rot, 1:3, μ; n_tol
                )
                @test ρ_rot isa Hermitian{ComplexF64, Matrix{ComplexF64}}
                @test ρ_rot ≈ ρ_rot_ref atol = 4 * eps()
                ρ_rot_31 = density_matrix_local(
                    H_rot, Σ_stat_rot, Σ_dyn_rot, 1:3, μ; orbitals = [3, 1], n_tol
                )
                @test ρ_rot_31 ≈ ρ_rot_ref[[3, 1], [3, 1]] atol = 4 * eps()
            end

            # a complex `Σ_dyn` alone makes the result complex
            Σ_dyn_c = PolesSumBlock(locs, [ComplexF64.(diagm(wgt[idx13])) for wgt in wgts])
            ρ_c = @inferred density_matrix_local(H_ks, Σ_stat, Σ_dyn_c, idx13, 0.75; n_tol)
            @test ρ_c isa Hermitian{ComplexF64, Matrix{ComplexF64}}
            ρ_ref = Diagonal(fillings_exact(0.75, H_ks_diag, Σ_stat_diag, locs, wgts))
            @test ρ_c ≈ ρ_ref atol = 4 * eps()
        end # basis and ordering
    end # density_matrix_local

    @testset "_issorted_and_unique" begin
        @test RAS_DMFT._issorted_and_unique(1:10)
        @test_throws ArgumentError RAS_DMFT._issorted_and_unique([1, 0]) # not sorted
        @test_throws ArgumentError RAS_DMFT._issorted_and_unique([0, 0, 1]) # not unique
        # duplicate zeros
        @test_throws ArgumentError RAS_DMFT._issorted_and_unique([-0.0, 0.0])
    end # _issorted_and_unique


end # util
