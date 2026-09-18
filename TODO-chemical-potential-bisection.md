# Chemical potential bisection

Handoff notes for the test in `test/utility.jl`, testset `find chemical potential`.
Written 2026-09-18 so the work can resume without the originating conversation.
Delete this file before opening the pull request.

## What the code under test does

`find_chemical_potential` (`src/utility.jl:112`) bisects `μ` until the filling matches
`n_fill`.

- Defaults: `μ_tol = 1.0e-6`, `b_max = 30`,
  `μ_min` and `μ_max` taken from the extreme locations of `Σ_dyn`, and `tol_weight = 0`.
- It builds the block arrowhead once (`src/utility.jl:134`),
  checks that `n_fill` is bracketed, then bisects (`src/utility.jl:149`).
- Tests that want more than six digits must override `μ_tol` and `b_max`.
  The test uses `μ_tol = 1.0e-13, b_max = 200`;
  with the defaults the accuracy caps at `1e-6` and any error below that is invisible.

`_filling_mu` (`src/utility.jl:178`) is the filling at fixed `μ`.

- It loops over k-points with `Threads.@threads`, accumulates atomically,
  and divides by `length(H_k)` at the end (`src/utility.jl:202`).
- `_arrowhead_eigen` (`src/utility.jl:163`) copies `Σ_A`
  and overwrites the top-left `n_b × n_b` block with `H + Σ_stat - μ * I`.
  Only the band block is shifted, because the self-energy poles sit at absolute
  frequencies.
- Each eigenvalue below the Fermi level contributes the squared norm of the band part of
  its eigenvector, not 1.
  A level within `tol = 1.0e-8` of the Fermi level contributes half.

## The math the test rests on

The eigenvalues of the arrowhead are the poles of

```
G(ω; μ)^(-1) = ω - (H_k + Σ_stat - μ) - Σ_dyn(ω)
```

and the residue at each pole is the outer product of the band part of the eigenvector.
So the filling is a weighted count, and a test that only fixes eigenvalue positions never
touches the weights.

If every pole weight is supported on a single band, and `H_k` and `Σ_stat` are diagonal in
that same basis, the arrowhead falls apart into independent blocks

```
[ d   √w ]          d = ϵ_b + σ_b - μ
[ √w  p  ]
```

with

```
λ± = (d + p ± sqrt((d - p)^2 + 4w)) / 2
‖v_band‖² = (p - λ)^2 / ((p - λ)^2 + w)
```

The two roots of a block share exactly one unit of band weight,
which follows from `(p - λ₊)(p - λ₋) = -w`.

Conjugating by `U ⊕ 1` with `U` unitary on the band block changes neither the eigenvalues
nor `‖v_band‖`, because `U 1 U' = 1` leaves the `μ` shift alone
and `‖U v_band‖ = ‖v_band‖`.
That is why the test builds everything in the decoupled frame and then rotates with
`U = Matrix(qr([...]).Q)`:
the reference stays exact while `H_k`, `Σ_stat` and the weights all turn dense.

`qr` is only a deterministic way to get an orthogonal matrix without `Random`.
Any fixed unitary works, as long as it acts on the band block alone
and is the same for every k-point,
since `Σ_stat` and `Σ_dyn` are k-independent and rotated once.

## What the test asserts

1. `_filling_mu` against the closed form at `μ ∈ {-2, -0.5, 0, 0.75, 1.7, 3}`,
   `atol = 8 * eps()`.
2. `find_chemical_potential` recovers `μ ∈ {-0.5, 0.75, 1.7}` from the exact filling,
   `atol = 1.0e-12`.
   The fillings are hardcoded literals, and a separate assertion pins them to the closed
   form.
3. A single block made singular at `μ = ϵ + σ - w / p` puts a level exactly at the Fermi
   level, where the band weight is `p² / (p² + w)` and half of it counts.
   This covers the `elseif ϵ <= tol` branch with an exact number instead of a symmetry
   argument.

The construction is 2 k-points with different `ϵ_k`, 3 bands, and 2 poles,
the first of which carries two bands and therefore has a rank-2 weight matrix.

## Traps

**Do not tidy the sampled `μ` values.**
The original sample at `μ = 0.8` disagreed with the closed form by `0.104`.
Cause: with `p = 0.5, ϵ = 1.1, σ = 0.4, w = 0.35` the block is singular exactly there,
since `d·p = w` gives `d = 0.7` and hence `μ = 0.8`.
The code half-counts that level while the reference tests `λ < 0` strictly,
so the gap is exactly `(1/N_k)·(1/2)·p²/(p²+w) = 0.104`.
The sample moved to `0.75`, where `min|λ| = 2.0e-2`, six orders above `tol = 1e-8`.
A comment in the test records this.

**The arrowhead is currently 9×9 where 6×6 is correct.**
The weights have ranks `[2, 1]` on 3 bands, so 6 is right,
but the test passes `tol_weight = 0`, which keeps three pure-noise directions.
The filling is unaffected, agreeing to `4e-16`,
which is itself evidence that noise directions carry no band weight.
See `TODO-amplitude-tolerance.md` for the fix.

## Why the old seeded test was dropped

It built random matrices under `Random.seed!(0)` and asserted
`μ ≈ 5.823782742023468`.
That number is a recorded output, so it detects change rather than correctness
and would have enshrined a bug just as happily.
`using Random` went with it, since nothing else in the file draws random numbers.

Coverage that went with it, and has since been restored by the rotation:
dense `H_k`, dense `Σ_stat`, a rank-2 weight matrix, and the k-average.

## Evidence the test bites

Mutating `_filling_mu` and restoring from a copy afterwards:

| mutation                                               | failures |
| ------------------------------------------------------ | -------- |
| `sum(abs2, v)` → `one(z)` below the Fermi level        | 9        |
| half-count at the Fermi level replaced by a full count | 2        |

## Left to do

1. Rebase onto `amplitude-tolerance` once that branch exists.
2. Pass `tol_weight` as a keyword instead of the positional `0.0`
   at `test/utility.jl:55`, `:84` and `:130`.
3. Add `@test size(Σ_A, 1) == 6`.
   That assertion is the point of the tolerance work and cannot be written before it.
4. Delete this file and reword the commit header without `WIP:`.

## Checks

```sh
runic --check .
cd test && julia --project=. -e 'include("utility.jl")'   # 30 tests
cd test && julia --project=. runtests.jl                  # 1033 tests
```
