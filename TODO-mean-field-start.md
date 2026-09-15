# TODO: use the mean-field ground state as initial guess

## Problem

`Wavefunction_singlet` and `RASWavefunction_singlet` (`src/wavefunctions.jl`)
start from the singlet of impurity `i` and mirror site `b`:

```julia
s1 = slater_start(K, 0b0110, L_v, L_c, V_v, V_c)   # i↑, b↓
s2 = slater_start(K, 0b1001, L_v, L_c, V_v, V_c)   # b↑, i↓
# both with amplitude 1/√2
```

That is only two of the four determinants of the mean-field ground state and
weights them equally, independent of the actual basis.

## Target

The ground state of the non-interacting reference system is
Eq. (E8) of Lu et al., PRB **90**, 085102 (2014):

```math
|ψ_0⟩ = (α a^†_{i↑} + β a^†_{b↑}) (α a^†_{i↓} + β a^†_{b↓})
        ∏_{j=1}^{N_v} a^†_{v_j↑} a^†_{v_j↓} |0⟩
```

with `α, β > 0` and `α² + β² = 1`.
In our notation `α = a_v` and `β = a_c`,
the amplitudes of the impurity in the occupied and empty subspace.

Expanding the spin product gives four determinants:

| `slater_start` pattern | state               | amplitude |
| ---------------------- | ------------------- | --------- |
| `0b0101`               | `i` doubly occupied | `a_v²`    |
| `0b0110`               | `i↑`, `b↓`          | `a_v a_c` |
| `0b1001`               | `b↑`, `i↓`          | `a_v a_c` |
| `0b1010`               | `b` doubly occupied | `a_c²`    |

The two cross terms form the singlet,
since `a^†_{b↑} a^†_{i↓} = -a^†_{i↓} a^†_{b↑}`.
The current guess keeps exactly those two and drops the double occupancies.

## Getting `a_v` and `a_c`

They are not stored in `NaturalImpurityOrbital`,
but `H_ib = R' diag(e_v¹, e_c¹) R` with `R = [a_v a_c; a_c -a_v]`,
so the eigenvector of the negative eigenvalue is `(a_v, a_c)`:

```julia
F = eigen(Symmetric(H_nat.H_ib))
a_v, a_c = F.vectors[:, 1]              # lowest eigenvalue is occupied
a_v < 0 && ((a_v, a_c) = (-a_v, -a_c))  # our construction has both ≥ 0
```

LAPACK fixes eigenvectors only up to a sign,
hence the explicit convention.

Decide then whether to store `a_v`, `a_c` as fields instead.
Deriving costs one 2×2 `eigen` per call,
storing adds two redundant numbers that can contradict `H_ib`.

## Touching

- `Wavefunction_singlet`, `RASWavefunction_singlet` in `src/wavefunctions.jl`
  need `a_v`, `a_c` (or the whole `NaturalImpurityOrbital`) as argument.
- `init_system` in `src/utility.jl` builds `ψ_start` and has `H_nat` at hand.
- `test/wavefunctions.jl` hardcodes ground-state energies and variances;
  a better guess changes the number of Lanczos steps to convergence.
