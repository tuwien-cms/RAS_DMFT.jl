# TODO: analyze full reorthogonalization in block Lanczos

`block_lanczos` (used by `correlator`) does not reorthogonalize,
while `block_lanczos_full_ortho` (used by the pole conversions) does.
Whether the correlator should pay that cost is open.

## What the difference causes

Without reorthogonalization an exhausted Krylov space is invisible.
Measured on `Δ = PolesSum([-1, 1], [0.5, 0.5])`, `U = 4`, `L_v = L_c = 0`, `p = 2`:

| `n_kryl` | poles | relative error of `G(0.7 + 0.3i)` |
| -------- | ----- | --------------------------------- |
| 3        | 6     | 3.5e-01                           |
| 5        | 10    | 2.4e-01                           |
| 10       | 20    | 3.7e-04                           |
| 20       | 31    | 0                                 |
| 100      | 31    | 0                                 |

The space is exhausted around `n_kryl = 20`,
yet `norm(B)` is still `1.03` at step 100 instead of dropping to zero,
so the early stop of `block_lanczos_full_ortho` cannot trigger here.
The vectors drift on roundoff rather than collapsing.

## Why it is not urgent

The surplus costs time, not accuracy:
the weights sum to `1.000000000000` at every `n_kryl`,
and the pole count saturates because `PolesSum` collapses the redundancy.
On a larger bath (11 poles, `L_v = L_c = 1`, `p = 2`) no breakdown occurs at all
within 200 steps, only ghost poles with weights down to 1e-43.

## Questions to answer

- Does reorthogonalization let `correlator` use a smaller `n_kryl`,
  paying `O(n_kryl)` inner products per step to save steps?
- Are the ghost poles ever large enough to matter downstream,
  or does `merge_small_weight!` always remove them?
- Is a cheaper partial reorthogonalization enough to make `norm(B)` collapse?
