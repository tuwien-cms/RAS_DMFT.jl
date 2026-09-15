# Physics Review of RAS_DMFT.jl

Review of the physics in the code, started 2026-09-15 at commit `69141e8`.
Findings are appended as they are found, with a severity tag:

- **[BUG]**: the code computes something different from what the physics says.
- **[SUSPECT]**: likely wrong or fragile, but not confirmed.
- **[LIMITATION]**: correct within stated assumptions, but the assumption is silent or narrow.
- **[NOTE]**: observation, no action required.

The scope is the single-band Anderson impurity model at arbitrary filling
and zero temperature.
The quasiparticle-weight regularization (`src/quasiparticle_weight.jl`) is excluded.

## Findings

### 1. [LIMITATION] The ground-state search is confined to one particle-number sector

`init_system` (`src/utility.jl:32`) builds the start state with every valence site doubly
occupied and one electron per spin in the impurity/mirror pair,
so the Lanczos ground state lives in the sector

- `N = 2 (n_v + 1)` electrons, where `n_v` is the number of negative eigenvalues of the
  mean-field arrowhead matrix in `natural_impurity_orbital` (`src/natural_impurity_orbital.jl:172`),
- `S_z = 0`.

`ground_state!` conserves both, and nothing afterwards checks whether a neighbouring sector
has lower energy.
At `T = 0` the impurity model is grand canonical (the Fermi level is at zero energy),
so the physical ground state is the minimum over all `N`.
Two consequences:

- The sector is decided by `ϵ_mf`, not by the physics.
  Full exact diagonalization of a 4-pole bath with `U = 2` (script `exp1` below) gives,
  for `ϵ_imp = -1.0`: `ϵ_mf = ϵ_imp` picks `N = 6` with `E = -6.726`,
  `ϵ_mf = ϵ_imp + U` picks `N = 4` with `E = -6.438`,
  and the impurity occupation jumps from 1.41 to 0.56 between the two.
  The same happens for `ϵ_imp = -0.5` (`n_imp` 1.18 vs 0.24).
- `N` is always even, but the global ground state is frequently in an odd sector.
  In the same 4-pole example the global minimum is the `N = 5`, `S_z = ±1/2` doublet
  for `ϵ_imp ∈ {-1.0, -0.5, -0.2}`, up to 0.6 below the even-sector energies.
  For a large bath the energy difference shrinks to the bath level spacing near the Fermi
  level, but the paramagnetic Green's function of a doublet requires spin averaging,
  which the current single spin-↓ correlator does not do.

There is already an indirect diagnostic: `_warn_wrong_sign` (`src/correlator.jl:137`)
warns when `C⁺` has weight at negative frequencies, which in exact arithmetic means
that the `N ± 1` sector reached by `d†ψ0` (or `dψ0`) has states below `E0`,
i.e. the chosen sector is not the ground state.
The code only warns and continues with a non-causal Green's function.

Suggested fix: after `natural_impurity_orbital`, run the ground-state search in the
sectors `N ∈ {2n_v, 2n_v + 2}` (singlet starts) and `N = 2n_v ± 1` (`S_z = 1/2` start)
and keep the lowest, or at least turn the wrong-sign warning into an error.
Both pieces already exist (`exp7`, 9-pole bath, `U = 2`, `ϵ_imp = -0.4`,
Hartree `ϵ_mf`, L=1, p=2):

- In the even sector the code returns `E0 = -4.739670`;
  `correlator_minus` with `d` then carries weight 0.109 at positive frequencies,
  and its lowest pole sits at `+0.0195 = E0 - E(N=9)`, i.e. the removal spectrum
  directly measures how far the true ground state lies below.
- A start determinant `slater_start(K, 0b0001, L_v, L_c, 0, 0)` (one electron in the
  impurity/mirror pair, `S_z = 1/2`) fed to `ground_state!` gives `E0 = -4.759171`
  against the exact `-4.759178` and the exact impurity occupation 0.8088.
  So odd sectors need no new infrastructure, only a different bit pattern than the
  hard-coded `0b0110`/`0b1001` in `Wavefunction_singlet`.

Verified facts from `exp1` (`U = 2`, 4-pole bath, full Fock space of 10 spin-orbitals):

- `Wavefunction_singlet` is a spin singlet (`⟨S²⟩ = 0`) and has the intended `N`.
- Lanczos from it reaches the lowest singlet of its sector to `1e-14`.
- The full one-particle spectrum of `Matrix(H_nat)` does not depend on `ϵ_mf`
  (max deviation `7e-14`), so the basis change itself is a correct unitary.

### 2. [LIMITATION] `ϵ_mf` is a free parameter with no prescription

Appendix E of Lu et al. (2014) defines the reference potential `V` (our `ϵ_mf`) such that
"the correct impurity occupation is reproduced", taken from the previous DMFT
iteration or a previous smaller-basis calculation.
Lu et al. 2019 use a Hartree-Fock solution.
`RAS_DMFT.jl` offers neither: `init_system` takes `ϵ_mf` as a number and the tutorial
passes `0`.
The user must know the impurity occupation to choose it.
Two options, both cheap because they only need eigenvalues of the arrowhead matrix:

- `ϵ_mf = ϵ_imp + U ⟨n_σ⟩` with `⟨n_σ⟩` self-consistent at Hartree level.
- bisection on `ϵ_mf` so that the mean-field impurity occupation
  `Σ_{ϵ_k<0} |⟨i|k⟩|²` equals the target `⟨n_σ⟩` from the last DMFT step
  (this is the 2014 prescription and what Quanty's `...Occupation` variant does).

See item 3 for how much this matters numerically.

### 3. [SUSPECT] Truncation error depends strongly on `ϵ_mf`; the bare level is the worst choice

Exact diagonalization of a 7-pole asymmetric bath (`exp2`, `U = 2`, `ϵ_imp = -1`,
sector `N = 8` for every choice) against `init_system` with `L_v = L_c = L`:

| `ϵ_mf`                  | L=1 p=1 ΔE | L=1 p=2 ΔE | L=2 p=1 ΔE | L=2 p=2 ΔE |
| ----------------------- | ---------- | ---------- | ---------- | ---------- |
| bare `ϵ_imp` (old code) | 3.5e-4     | 2.4e-6     | 3.5e-6     | 1.9e-9     |
| `ϵ_imp + U/2`           | 2.5e-5     | 4.0e-7     | 1.6e-7     | 1.8e-10    |
| Hartree self-consistent | 1.6e-5     | 4.6e-7     | 7.2e-8     | 2.1e-10    |
| occupation-matched      | 1.5e-5     | 4.7e-7     | 6.5e-8     | 2.1e-10    |

The impurity occupation and double occupancy errors follow the same pattern
(`Δn` 2e-4 vs 1e-5 at L=1, p=1).
A 9-pole bath (`exp2b`) reproduces this for `ϵ_imp ∈ {-1.0, -0.4, -1.5}`:
bare `ϵ_mf` gives `ΔE ≈ 8e-4` at L=1, p=1, the Hartree level `3e-5`,
and the ratio stays near 10 for L=2, p=2.
So the mean-field level that was "coded wrong for years" costs one to two orders of
magnitude in accuracy at fixed `(L, p)`, without changing the sector in these cases.
Hartree-self-consistent and occupation-matched levels are equivalent;
either is a good default (see item 2).
At p=2 the plain `ϵ_imp + U/2` sometimes beats the Hartree level
(`4e-8` vs `8e-7` for `ϵ_imp = -1.5`, L=1),
so no closed formula is optimal.
Because the RAS energy in a fixed sector is variational,
`E0(ϵ_mf)` can simply be minimized over `ϵ_mf` (one-dimensional, cheap at L=1),
which also removes the guesswork.

For `ϵ_imp = -0.4` the two even sectors reachable by different `ϵ_mf` have
`E(N=8) = -3.7920` (bare) and `E(N=6) = -3.7713` (`ϵ_imp + U/2`),
i.e. here the Hartree-like level lands in the higher sector.
This is item 1 again: the sector must be chosen by energy, not by `ϵ_mf`.

### 4. [BUG] `natural_impurity_orbital_ras_operator` fails when a chain fits in the bit part

`src/natural_impurity_orbital.jl:372` reads `H_nat.t_c[n_c_bit]` (and line 369
`H_nat.t_v[n_v_bit]`) for the hopping into the vector part.
When `n_c_bit == n_c` (the whole conduction chain is inside the bit component,
allowed by the check `n_c_bit <= n_c` on line 349) the hopping does not exist and
the call throws `BoundsError`.
Physically the mixed hopping is simply absent; the fix is to add nothing (or zero
amplitude) in that case, or to require `n_c_bit < n_c` explicitly.
Hit in `exp2` with a 7-pole bath, `ϵ_mf = 0.6`, `n_c = 2`, `L_c = 2`.

### 5. [NOTE] Spectra are far less sensitive to `ϵ_mf` than the ground-state energy

`exp3` compares the RAS impurity Green's function and the IFG self-energy with the exact
Lehmann result of the same 7-pole bath (`U = 2`, Lorentzian broadening 0.1,
relative maximum deviation of the spectral function `A` and of `Im Σ`):

| `ϵ_imp` | `ϵ_mf`  | L=1 p=1 A / ImΣ | L=1 p=2 A / ImΣ | L=2 p=2 A / ImΣ |
| ------- | ------- | --------------- | --------------- | --------------- |
| -1.0    | bare    | 13 % / 17 %     | 5.9 % / 15 %    | 2.0 % / 4.0 %   |
| -1.0    | Hartree | 13 % / 15 %     | 5.1 % / 12 %    | 1.9 % / 3.0 %   |
| -0.4    | bare    | 15 % / 16 %     | 7.4 % / 7.3 %   | 1.8 % / 5.2 %   |
| -0.4    | Hartree | 14 % / 14 %     | 6.3 % / 6.2 %   | 1.7 % / 5.4 %   |

While the ground-state energy error drops by a factor 20 with the Hartree level,
the spectra improve by at most 20 %.
The Green's function error is dominated by the truncation of the excited-state space
`(L, p)` and by the finite Krylov space, not by the reference level.
So the old wrong `ϵ_mf` mostly harmed ground-state observables
(energy, occupation, double occupancy, hence `Σ_H`) and the sector selection,
not the line shape at fixed sector.

Sum rules hold in every run to the ground-state convergence
(`var = 1e-12`, hence `1e-7` in the moments):
`M0(G) = 1`, `M1(G) = ϵ_imp + Σ_H`, `M2(G) = ϵ_imp² + 2ϵ_imp Σ_H + U²⟨n↑⟩ + M0(Δ)`,
`M0(Σ_dyn) = U² ⟨n↑⟩(1 - ⟨n↑⟩)` to `1e-14`.
No wrong-sign poles appeared.

### 6. [NOTE] The DMFT self-consistency pieces are algebraically correct

Verified numerically (`exp5`) to machine precision:

- `update_hybridization_function`: `Δ_new(z) = Δ0(z + μ - Σ_H - Σ(z))` at complex `z`
  (`2e-15`), weight conserved.
  It is the Bethe-lattice relation only; the docstring says so.
- `inverse`: `1/G(z) = z - a0 - D(z)` (`3e-14`).
- `find_chemical_potential` + `greens_function_local`: the returned filling equals
  `Tr ∫_{-∞}^{0} A` of the returned `G_loc`, total weight equals the number of bands,
  and `G_loc(z)` equals the direct `k`-sum (`1e-15`).
- `self_energy_IFG`: on a synthetic exact block correlator with known `Σ_dyn`
  the Schur complement recovers `Σ_dyn(z)` to `7e-15`;
  the extra poles carry exactly zero weight.
  The algebra `Σ_dyn = Ĩ - F̃ G⁻¹ F̃†` with `q̃ = q - Σ_H d` is right,
  and `M0(Σ_dyn) = U²⟨n↑⟩(1-⟨n↑⟩)` follows from `⟨{q̃, d†}⟩ = 0`.
- `Wavefunction_singlet`: `⟨S²⟩ = 0`, and the block correlator assembly
  `transpose(C⁻) + C⁺` gives `G_ij = ⟨⟨a_i; a_j†⟩⟩` with `a = (q̃, d)`.
- The natural-orbital couplings satisfy Lu 2014 Eq. (E9),
  `α t_ic + β t_bc = 0` and `-β t_iv + α t_bv = 0`, by construction
  (`i_c = a_c t_c`, `b_c = -a_v t_c`, `i_v = a_v t_v`, `b_v = a_c t_v`).
- Bethe discretizations reproduce the semicircle moments
  (`M2 = D²/4`, `M4 = D⁴/8`) up to the expected discretization error;
  `greens_function_bethe_equal_weight(51)` is off by `4e-3` in `M2` because the pole
  locations come from a 128-point trapezoid rule, which is coarse but harmless.

### 7. [SUSPECT] Usage in `CeRu4Sn6/scripts/dmft_loop.jl`

Read-only observations on the driver that consumes this package:

- It still calls the seven-argument `init_system(Δ, H_int, ϵ_imp, L_v, L_c, p, var_gs)`,
  i.e. the reference level equals the bare `ϵ_imp = a0 - Σ_H`.
  For a Ce 4f level several eV below the Fermi energy this puts the mean-field
  impurity level far below `E_F`, so the reference has the impurity nearly doubly
  occupied and, by interlacing, may count one more occupied level than the Hartree
  reference (items 1 and 3).
  The natural replacement is `ϵ_mf = a0` (the first moment of `G_imp`,
  `= ϵ_imp + Σ_H`), which is the Hartree reference, or an occupation-matched level.
- The two even sectors `N` and `N ± 2` are separated by the level spacing of the
  log grid near `E_F` (0.3 meV), so a wrong sector costs almost no energy but changes
  the impurity occupation by up to one electron spread over the bath.
  Compare `occ_up + occ_dn` from the ground state with `filling(G_imp)` from the
  previous `G_loc` as a cheap consistency check each iteration.
- `Δ = P1 - Σ_imp_dyn` followed by `merge_negative_weight!` is the standard
  real-axis approximation; its error is not controlled and grows when `Σ_dyn` has
  poles far from those of `P1`.
  This is a property of the method, not a bug.

### 8. [SUSPECT] Sharing a zero-energy reference level between both chains is costly

`natural_impurity_orbital` (`src/natural_impurity_orbital.jl:180`) gives a reference
eigenstate at the Fermi energy to both chains with half its weight.
This happens whenever the mean-field arrowhead matrix has a zero eigenvalue,
in practice at particle-hole symmetry with an even number of hybridization poles
(odd arrowhead dimension).
Consequences, all verified in `exp4b` (`U = 2`, `ϵ_imp = -1`, PHS baths):

- The single-particle basis has one orbital more than the star geometry
  (`size(H_nat, 1) == length(Δ) + 2`).
  The extra orbital is an exactly decoupled zero mode
  (`(|0_v⟩ - |0_c⟩)/√2` has no coupling to the impurity), see also the note
  "the redundancy is harmless" in `natural-impurity-orbitals.md`.
- Because that mode is free, the many-body ground state in the code's sector is
  two-fold degenerate and coincides with the physical odd-`N` doublet of the star
  geometry (`E = -7.1530` for 4 poles, equal to the star `N = 5` energy).
  Lanczos from the singlet start converges to the spin-symmetric combination,
  whose spin-↓ Green's function equals the spin-averaged doublet Green's function
  to `1e-12`.
  So the result is physically right in the full space; item 1 is "accidentally" solved
  for this case.
- The RAS truncation suffers badly, because emptying half of the decoupled mode
  requires holes deep in the valence chain:

  | bath            | L=1 p=1 ΔE | L=1 p=2 ΔE | L=2 p=1 ΔE | L=2 p=2 ΔE |
  | --------------- | ---------- | ---------- | ---------- | ---------- |
  | 4 poles (split) | 1.3e-3     | 2.2e-5     |            |            |
  | 5 poles         | 2.3e-5     | 2.6e-7     |            |            |
  | 6 poles (split) | 2.0e-2     | 2.7e-3     | 9.6e-4     | 1.4e-5     |
  | 7 poles         | 9.1e-5     | 1.2e-6     | 5.4e-7     | 6.5e-10    |

  A bath one pole larger converges 2 to 4 orders of magnitude better as soon as the
  zero mode is absent.

Recommendation: never feed an even number of particle-hole-symmetric poles
(for example `range(-D, D; length = even)` after `remove_zero_weight!`) into the solver at
half filling; document it, or assign a zero-energy reference level entirely to one
chain (which breaks the symmetry of the basis but keeps the Hilbert space size).
The tutorial (`N = 51` including `0`) and the tests (`n_bath = 31, 101`) happen to
avoid the case; the "PHS insulator" test codifies the behaviour.

### 9. [NOTE] Smaller items checked and found correct

- `temperature_kondo` matches Haldane's formula for the asymmetric Anderson model,
  `T_K = sqrt(UΔ/2) exp(π ϵ (ϵ + U) / (2ΔU))`, with `Δ0` the hybridization half-width
  `π V² ρ(0) = -Im Δ(0)`.
  The docstring should say which `Δ0` is meant.
- `greens_function_bethe_analytic` picks the retarded branch on both sides of the
  band and inside it (`Im G < 0`, agrees with a 4001-pole sum to `1e-5`).
- `_gaussian_broadened` is the exact Hilbert transform of a Gaussian
  (Dawson-function form, agrees with numerical integration to `5e-4` at 200 001
  quadrature points).
- `natural_impurity_orbital_operator` with `n_v_bit ≠ n_c_bit` gives the same
  many-body energies as the symmetric orderings (`1e-14`), so the site bookkeeping
  for the reordered `[i, b, v_1..v_L, c_1..c_L, v_rest, c_rest]` layout is right.
- The full test suite passes on this machine (983 tests, Julia 1.13).

### 10. [NOTE] Lanczos without reorthogonalization (Fermions.jl)

`Fermions.Lanczos.lanczos`, `Fermions.Lanczos.block_lanczos` and the local
`RAS_DMFT.block_lanczos` (`src/block_lanczos.jl`) use the plain three-term recursion.
For `n_kryl = 50…100` and the Green's function this is the usual practice:
loss of orthogonality produces ghost copies of converged poles,
which split weight between duplicates but leave the continued fraction a valid
approximant (moments up to order `2 n_kryl` stay exact).
It does mean that `length(C) == 2 n_kryl` poles are not `2 n_kryl` independent
excitations, and that `merge_degenerate_poles!`/`merge_small_weight!` are doing
real work after every correlator call.
`ground_state!` restarts every 5 steps and renormalizes, so it is not affected.
No action needed unless `n_kryl` is pushed to several hundred as in Lu 2019
(`M = 300…400`), where full reorthogonalization (as already done in
`block_lanczos_full_ortho` for the pole conversions) becomes advisable.

## Summary

Ranked by expected impact on production results:

1. **Sector selection (items 1, 2, 3).**
   `ϵ_mf` silently decides the electron number of the impurity model and the code never
   compares neighbouring sectors.
   With the old bare `ϵ_mf` the ground-state error at fixed `(L, p)` is 10 to 25 times
   larger than with a Hartree or occupation-matched level,
   and away from half filling either choice can land in a sector that is not the
   ground state.
   Fix: choose `ϵ_mf` from the target occupation (or minimize `E0(ϵ_mf)`),
   then compare `E0` across `N = 2n_v - 1, 2n_v, 2n_v + 1, 2n_v + 2`,
   and make wrong-sign weight in `C⁺`/`C⁻` an error.
2. **Zero-mode sharing (item 8).**
   Costs 2 to 4 orders of magnitude in RAS accuracy at particle-hole symmetry with an
   even number of poles.
   Avoid the case or assign the level to one chain.
3. **`n_c_bit == n_c` crash (item 4).** Small code fix.
4. **CeRu4Sn6 driver (item 7).** Update to the new `init_system` signature with
   `ϵ_mf = a0` and log `occ_up + occ_dn` against `filling(G_imp)`.

Everything else that was checked (block correlator assembly, IFG Schur complement,
Bethe update, chemical potential search, local Green's function, natural-orbital
couplings, spin of the start state, sum rules, broadening, Kondo formula) is correct.
The impurity spectra at fixed sector are insensitive to `ϵ_mf` (item 5).

## Scripts

The numerical checks live in `physics-review-scripts/` (untracked):

| script                | content                                                                  |
| --------------------- | ------------------------------------------------------------------------ |
| `exp1.jl`             | full ED of a 4-pole bath: sectors, start-state spin, `ϵ_mf` independence |
| `exp2.jl`             | RAS vs exact, 7-pole bath, `ϵ_mf` scan (hits the item 4 crash)           |
| `exp2b.jl`            | RAS vs exact, 9-pole bath, even and odd sectors                          |
| `exp3.jl`             | RAS spectra and IFG self-energy vs exact Lehmann, sum rules              |
| `exp4.jl`, `exp4b.jl` | zero-mode sharing: Hilbert space, degeneracy, RAS accuracy               |
| `exp5.jl`             | DMFT pieces: Bethe update, `inverse`, `μ` search, `G_loc`, IFG, moments  |
| `exp6.jl`, `exp6b.jl` | asymmetric `L_v ≠ L_c`, analytic Bethe branch, broadening                |
| `exp7.jl`             | wrong-sector diagnostics and an `S_z = 1/2` start                        |

Run from the repository root with `julia --project=. -t 4 physics-review-scripts/expN.jl`.
