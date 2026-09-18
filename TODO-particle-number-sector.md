# TODO: document the fixed particle-number sector

Deferred from the bug audit of `69141e8`
because it is a design caveat rather than a coding error.

## Problem

`RASWavefunction_singlet` (`src/wavefunctions.jl`) starts from the impurity-mirror
singlet with all valence sites filled and all conduction sites empty,
so the start state carries

```math
N = 2 + 2 n_valence.
```

The Anderson Hamiltonian conserves particle number,
so every Krylov vector `H^k ψ` carries the same `N` as the start state.
Lanczos therefore cannot leave that sector,
and `ground_state!` returns the ground state _of that sector_,
not of the grand-canonical problem.
The same holds for the spin, as the start state is a singlet.

Measured on a Bethe bath with 11 poles:
`N = 12` for the start state and `12.00000000000003` after convergence.

## Why the sector can be wrong

`n_valence` counts the occupied _mean-field_ levels,
fixed by `ϵ_mf` in `natural_impurity_orbital` before the interaction enters.
For an asymmetric bath the count does move with `ϵ_mf`:

| `ϵ_mf`     | `n_valence` | `N` |
| ---------- | ----------- | --- |
| -3.0 … 1.0 | 2           | 6   |
| 3.0        | 1           | 4   |

A symmetric bath pins the count by symmetry,
which is why the particle-hole symmetric test cases never expose this.

DMFT asks for the ground state of `H - μN`,
so interactions can move the optimal `N` away from the mean-field count.
Exact diagonalization of the full Fock space (script C of the audit):

| sector | minimum energy |
| ------ | -------------- |
| N = 3  | -4.3348        |
| N = 4  | -5.0151 (RAS)  |
| N = 5  | -5.0711        |
| N = 6  | -4.7092        |

The reported ground state is then 0.056 above the true one, silently.

## Symptom and existing detection

A wrong sector makes `G^+(ω)` acquire poles at negative frequency,
because adding an electron lowers the energy,
which cannot happen for a true ground state.
`_warn_wrong_sign` (`src/correlator.jl`) already fires on exactly this condition.

## Suggested resolution

1. Document the constraint in `init_system` and `ground_state!`:
   `N` and `S` are fixed by the start state,
   `N` follows from `ϵ_mf`,
   and the result is the ground state of that sector.
2. Extend the `_warn_wrong_sign` message to name the likely cause,
   as it currently reports the symptom only.

The audit also suggests a Lanczos run from the `N ± 1` reference determinants,
but `correlator_plus` and `correlator_minus` already build those Krylov spaces,
so their lowest poles are that comparison and the extra run would duplicate it.

## Not a coding error

The natural impurity orbital basis is constructed from a definite filling:
the valence/conduction split is the occupied/empty split of the mean-field
density matrix, so varying `N` means rebuilding the basis.
Quanty behaves the same way.
