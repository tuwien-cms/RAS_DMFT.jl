using RAS_DMFT
using Fermions
using Fermions.Wavefunctions
using LinearAlgebra
using Test

@testset "wavefunctions" begin
    @testset "Wavefunction_singlet" begin
        ψ = Wavefunction_singlet(Dict{UInt64, Float64}, 1, 2, 3, 4)
        d = Dict(
            UInt64(0b0000_111_00_1_01_0000_111_00_1_10) => 1 / sqrt(2),
            UInt64(0b0000_111_00_1_10_0000_111_00_1_01) => 1 / sqrt(2),
        )
        ϕ = Wavefunction(d)
        @test ψ == ϕ
    end # Wavefunction_singlet

    @testset "RASWavefunction_singlet" begin
        # excitation = 0
        ψ = RASWavefunction_singlet(Dict{UInt64, Float64}, 1, 2, 3, 4, 0)
        # vectors must be the equal but no egal
        foo = collect(values(ψ))
        @test foo[1] == foo[2]
        @test foo[1] !== foo[2]
        v = [1 / sqrt(2)]
        d = Dict(UInt64(0b00_1_01_00_1_10) => copy(v), UInt64(0b00_1_10_00_1_01) => copy(v))
        ϕ = RASWavefunction(d, 5, 3, 4, 0)
        @test ψ == ϕ

        # excitation = 1
        ψ = RASWavefunction_singlet(Dict{UInt64, Float64}, 1, 2, 3, 4, 1)
        v = zeros(1 + 2 * (3 + 4))
        v[1] = 1 / sqrt(2)
        d = Dict(UInt64(0b00_1_01_00_1_10) => copy(v), UInt64(0b00_1_10_00_1_01) => copy(v))
        ϕ = RASWavefunction(d, 5, 3, 4, 1)
        @test ψ == ϕ

        # excitation = 2
        ψ = RASWavefunction_singlet(Dict{UInt64, Float64}, 1, 2, 3, 4, 2)
        v = zeros(1 + 14 + 14 * 13 ÷ 2)
        v[1] = 1 / sqrt(2)
        d = Dict(UInt64(0b00_1_01_00_1_10) => copy(v), UInt64(0b00_1_10_00_1_01) => copy(v))
        ϕ = RASWavefunction(d, 5, 3, 4, 2)
        @test ψ == ϕ
    end # RASWavefunction_singlet

    @testset "ground state" begin
        # parameters
        n_bath = 31
        U = 4.0
        μ = U / 2
        L_v = 1
        L_c = 1
        p = 2
        n_kryl = 5

        Δ = hybridization_function_bethe_simple(n_bath)
        H_nat = natural_impurity_orbital(Δ, 0)

        # RASWavefunction
        fs = FockSpace(Orbitals(2 + L_v + L_c), FermionicSpin(1 // 2))
        n = occupations(fs)
        H_int = U * n[1, -1 // 2] * n[1, 1 // 2]
        H = natural_impurity_orbital_ras_operator(H_nat, H_int, -μ, fs, L_v, L_c, p)
        ψ_start = RASWavefunction_singlet(
            Dict{UInt64, Float64}, L_v, L_c, H.nfilled, H.nempty, p
        )

        # Every restart improves the state.
        ψ0 = ψ_start
        E = Vector{Float64}(undef, 3)
        var = Vector{Float64}(undef, 3)
        for i in eachindex(E)
            # a variance of 0 is never reached, so every call warns
            E[i], ψ0 = @test_logs (:warn,) ground_state!(H, ψ0, n_kryl, 10, 0)
            foo = H * ψ0
            var[i] = foo ⋅ foo
        end

        # the energy only decreases, by an ever smaller amount
        @test all(<(0), E)
        @test abs(E[3]) < abs(E[2]) < abs(E[1])
        # the state approaches an eigenstate
        @test var[3] < var[2] < var[1]

        # energy and variance after 10, 20, and 30 Krylov steps
        E_total = cumsum(E)
        @test E_total[1] ≈ -21.52794995443258 rtol = 2.0e-13
        @test E_total[2] ≈ -21.527949990414943 rtol = 2.0e-13
        @test E_total[3] ≈ -21.527949990415216 rtol = 2.0e-13
        @test var[1] < 3.0e-8
        @test var[2] < 3.0e-13
        @test var[3] < eps()

        # variance on unrestricted space is worse
        fs = FockSpace(Orbitals(size(H_nat, 1)), FermionicSpin(1 // 2))
        n = occupations(fs)
        H_int = U * n[1, -1 // 2] * n[1, 1 // 2]
        H_wf = natural_impurity_orbital_operator(H_nat, H_int, -μ, fs, L_v, L_c)
        Fermions.shift_spectrum!(H_wf, E_total[3])
        ψ0_wf = Wavefunction(ψ0)
        foo = H_wf * ψ0_wf
        var = foo ⋅ foo
        @test 1.0e-4 < var < 2.0e-4

        # symmetric interaction U (n_↑ - 1/2) (n_↓ - 1/2)
        # with `ϵ_imp = 0` must raise the eigenenergy by U/4
        fs = FockSpace(Orbitals(2 + L_v + L_c), FermionicSpin(1 // 2))
        n = occupations(fs)
        H_int = U * (n[1, -1 // 2] - 0.5 * I) * (n[1, 1 // 2] - 0.5 * I)
        H = natural_impurity_orbital_ras_operator(H_nat, H_int, 0.0, fs, L_v, L_c, p)
        ψ_start = RASWavefunction_singlet(
            Dict{UInt64, Float64}, L_v, L_c, H.nfilled, H.nempty, p
        )
        E_sym, _ = @test_logs (:warn,) ground_state!(H, ψ_start, n_kryl, 10, 0)
        @test E_sym - U / 4 ≈ E_total[1] rtol = 2.0e-13
    end # ground state
end # wavefunctions
