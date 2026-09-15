# Natural Impurity Orbitals

Notes compiled from the two Lu et al. papers, the Quanty source code,
the Quanty online documentation, and the DiscreteDMFT draft.

## References

- Y. Lu, M. Höppner, O. Gunnarsson, M. W. Haverkort,
  _Efficient real-frequency solver for dynamical mean-field theory_,
  Phys. Rev. B **90**, 085102 (2014).
  DOI: 10.1103/PhysRevB.90.085102.
  Construction: Appendices D and E.
  Zotero: `storage/LGFCJJC9`.
- Y. Lu, X. Cao, P. Hansmann, M. W. Haverkort,
  _Natural-orbital impurity solver and projection approach for Green's functions_,
  Phys. Rev. B **100**, 115134 (2019).
  DOI: 10.1103/PhysRevB.100.115134.
  Recipe: Sec. II A; projection: Secs. II B-C.
  Zotero: `storage/TQYQUHLE`.
- Quanty documentation, ResponseFunction object:
  <https://www.quanty.org/documentation/language_reference/objects/responsefunction/start>.
  Lists the four storage formats; format 4 is "Natural Impurity".
- Draft: Začinskis, Pelz, Ebel, Kugler, von Delft, Held, Haverkort, Gleis,
  _Discrete Green's function representation, self energy calculations and impurity solvers in DMFT_,
  `/home/frank/.local/state/temp/DiscreteDMFT.tex`
  (App. "Natural impurity-orbital representation").
- Quanty source: `/home/frank/data/projects/quanty`
  (all code paths below are relative to this repo).

## What Are Natural Impurity Orbitals?

Natural orbitals are the single-particle basis in which the ground-state
one-particle density matrix is diagonal.
They minimize the number of Slater determinants needed to represent
a correlated ground state (exactly one determinant at $U = 0$).

For an Anderson impurity model a naive application fails:
diagonalizing the density matrix of the _whole_ system mixes the interacting
impurity orbitals with the noninteracting bath orbitals,
turning the local interaction $H_\mathrm{loc}$ into long-range interactions
over all orbitals.

**Natural impurity orbitals** therefore restrict the basis rotation:
the impurity block and the bath block of the density matrix are rotated
separately and never mixed.
The resulting single-particle geometry is

- impurity site $i$ (occupation $n$),
- one special bath site $b$ (occupation $1 - n$),
  forming a "molecular bond" with $i$,
- a fully occupied **valence chain** $v_1, v_2, \ldots$,
- a completely empty **conduction chain** $c_1, c_2, \ldots$,

where $i$ and $b$ both couple to the first site of each chain.
A destructive-interference condition
$\alpha\, t_{ic} + \beta\, t_{bc} = 0$
decouples the occupied bond state from the conduction chain
(2014, App. E).

Properties (2019, Sec. III):

- Exact at $U \to 0$: the ground state needs only $2m$ determinants
  ($m$ = number of impurity spin orbitals; 4 for one band).
- Exact at $U \to \infty$ (Mott insulator): particle-hole excitations
  into the chains are suppressed by the gap $\approx U - 2D$.
- At finite $U$, electron (hole) leakage into the conduction (valence)
  chain decays (near-)exponentially with the site index;
  slowest for the correlated metal $U/D \approx 2$,
  but even there the density drops below $10^{-3}$ within 2-4 sites.
- Ground-state energy and Green's functions converge exponentially
  in the subsystem size $L$.

## How to Calculate Them

### Wave-function recipe (2019, Sec. II A)

1. Solve the impurity model in mean field (e.g. Hartree-Fock) and
   compute the single-particle density matrix
   $\hat\rho^\mathrm{MF} = (\hat\rho_i, \hat\rho_{il}; \hat\rho_{li}, \hat\rho_l)$.
2. Diagonalize only the bath block $\hat\rho_l^\mathrm{MF}$.
   All bath orbitals get occupation 0 or 1, except $m$ fractional ones,
   which define the site $b$
   (with $\operatorname{Tr}\hat\rho_b = m - \operatorname{Tr}\hat\rho_i$).
3. Combine $i$ and $b$ into bonding/antibonding orbitals with
   occupation $m$ and $0$;
   the mean-field Hamiltonian decouples into a filled and an empty sector.
4. Lanczos-tridiagonalize each sector into a chain
   (filled → valence, empty → conduction).
5. Reverse the rotation of step 3 to recover $i$ and $b$.

The 2014 paper phrases step 1-2 as a **noninteracting reference system**:
replace $U$ by a potential $V$ tuned to reproduce the correct impurity
occupation, solve the one-body problem,
and diagonalize the bath density matrix of that Slater determinant.

### Green's-function recipe (DiscreteDMFT draft, App. A.4; what Quanty does)

Work directly on the hybridization function as a list of poles,
$G_\mathrm{P}(\omega) = A_0 + \sum_i B_i (\omega - a_i + \mathrm{i}0^+)^{-1} B_i$
with Hermitian principal square roots $B_i = W_i^{1/2}$.

1. **Valence-conduction split** with the Fermi function $f$:
   $G_\mathrm{P}^v = \sum_i f(a_i) B_i (\omega - a_i)^{-1} B_i$,
   $G_\mathrm{P}^c = \sum_i (1 - f(a_i)) B_i (\omega - a_i)^{-1} B_i$.
   The temperature is an artificial smearing parameter.
2. **Tridiagonalize** $G^v$ and $G^c$ separately (block Lanczos).
   The prefactors are no longer unity, but satisfy
   $(B_0^v)^2 + (B_0^c)^2 = I_N$.
3. **Assemble** the decoupled Hamiltonian $H'_N$
   (valence chain and conduction chain, block-diagonal in the two sectors)
   and rotate only the first two $N \times N$ blocks:
   $H_N = U_N H'_N U_N$ with

   $$
   U_N =
   \begin{pmatrix}
   B_0^v & B_0^c \\
   B_0^c & -B_0^v
   \end{pmatrix} \oplus I .
   $$

   $B_0^c = (I - (B_0^v)^2)^{1/2}$ implies $[B_0^v, B_0^c] = 0$,
   so $U_N$ is unitary.

The docs' evaluation formula for the stored format is
$G(\omega,\Gamma) = A_0 + B_0^* (G_\mathrm{val} + G_\mathrm{con}) B_0^T$.

## Implementation in Quanty

Response functions are stored in four formats
(`ResponsefunctionType`, `Core/StructDef.cpp:835`):
`l` list of poles, `t` tridiagonal (chain), `a` Anderson (star),
`n` natural impurity orbital.

### Scalar form (single band, standalone DMFT solver)

Type: `BADoubleTriDiagonalMatrixType` ("bonding-antibonding"),
two scalar tridiagonal chains `val`/`con` plus the corner elements
`a0, b0, acvB, acvA, acvBA, bvB0, bvA0, bcB0, bcA0`.

- Split at the Fermi level:
  `ListOfPolesCopyVal/Con`
  (`Core/BasicMath/BasicMath_ListOfPoles.cpp:1460,1490`).
  Poles inside $[E_F - \varepsilon, E_F + \varepsilon]$ are shared
  between both parts with linear ramp weights
  (T = 0 analog of the Fermi split).
- Occupation-constrained variant:
  `ListOfPolesCopyVal/ConOccupation` (`:1521,1555`) and
  `ListOfPolesToBADoubleTridiagonalMatrixOccupation`
  (`Core/BasicMath/BasicMath_MatrixTypeConversion.cpp:4019`);
  fixes the total valence weight to a target occupation by partially
  including the boundary pole, then sets
  $\varepsilon = |\mu - \mu_\mathrm{occ}|$.
  Code-level realization of the 2014 "reference potential $V$".
- Full transformation:
  `ListOfPolesToBADoubleTridiagonalMatrix` (`:3987`) —
  split, tridiagonalize both parts, set the corner.
- Bonding/antibonding corner:
  `SetBADoubleTriDiagonalMatrixConVal`
  (`Core/BasicMath/BasicMath_BADoubleTriDiagonalMatrixVector.cpp:96`):

  ```c
  b0   = sqrt(bc0^2 + bv0^2);  bv0 /= b0;  bc0 /= b0;
  acvB  = bv0^2 av1 + bc0^2 ac1;
  acvA  = bv0^2 ac1 + bc0^2 av1;
  acvBA = bv0 bc0 (av1 - ac1);
  bvB0 = bv0 bv1;  bcB0 = bc0 bc1;
  bvA0 = bc0 bv1;  bcA0 = -bv0 bc1;   // interference cancellation
  ```

  The minus sign in `bcA0` is the
  $\alpha t_{ic} + \beta t_{bc} = 0$ cancellation.

DMFT usage (`DMFT/DMFT.cpp`, `DMFT/SingleSiteFunctions.cpp`):

- `createGBathBA` (`:299`): bath Green's function → BA double chain.
- `CreateWaveFunctionDMFT` (`:1`): reference determinant —
  valence chain filled, conduction chain empty
  (the $U \to 0$ ground state).
- `createGBathBATrunc` (`:329`): keep `NBath + 1` sites per chain
  → cluster $H_I$ (the projection at bond $L$).
- `createVPerturbation` (`:360`): hopping `b[NBath+1]` at the cut bond
  → operator $V$; discarded chain remainders → tridiagonal propagators
  `gPerturbation` (energies sign-flipped for the valence/hole part).
- `FindGroundStateandMu` (`:948`) → `LanczosGroundStateRR`
  (`Core/Lanczos.cpp:225`): adaptive determinant basis (2014, App. D) —
  apply $H$ to spawn determinants (`OperatorPsiRR`),
  drop weights below $\epsilon$ (`RealWaveFunctionTruncateBasis`),
  Lanczos in the fixed basis, repeat;
  $\epsilon$ doubles automatically on memory exhaustion.
- `CreateGclDMFT` (`:1117`) → `LanczosTriDiagonalizeDysonLowMem`
  (`Core/Lanczos.cpp:6034`): Green's function with the cut chains
  re-attached (2019, Sec. II C) —
  non-orthogonal extended basis $V_i |\mathrm{Krylov}_k\rangle$,
  overlap $S$ and Hamiltonian matrices, generalized diagonalization,
  Dyson recombination with the chain propagators
  (`TridiagonalizeImpurityCoupledToBathDyson`).

### Block form (multiorbital)

Type: `BlockNaturalImpurityOrbitalMatrixType`
(`Core/StructDef.cpp:819`):
`BlockSize` ($N$), block-tridiagonal chains `val`/`con`,
impurity block `a0` ($N \times N$),
corner matrix `TMT` ($4N \times 4N$), smearing `Epsilon`.
ASCII art of the matrices $A = T M T^C$ (natural geometry $A$,
decoupled form $M$, rotation $T$) in
`Core/BasicMathBlock/BasicMath_BlockNaturalImpurityOrbital.cpp:1-45`.

- Split with Fermi weights:
  `CopyBlockListOfPolesVal/Con`
  (`Core/BasicMathBlock/BasicMath_BlockListOfPoles.cpp:76,109`) —
  each pole's weight matrix multiplied by $f(a_k, \mu, T)$
  resp. $1 - f$; $T$ is the "effective artificial temperature".
- Full transformation:
  `BlockListOfPolesToBlockNaturalImpurityOrbitalMatrix`
  (`Core/BasicMathBlock/BasicMath_BlockTypeConversion.cpp:819`);
  `...ReduceDimension` variant (`:873`) additionally applies a
  rectangular impurity transformation `AT`.
  Also available: `BlockAndersonMatrixTo...` (`:1147`),
  `BlockTriDiagonalMatrixTo...` (`:2268`).
- Block tridiagonalization:
  `BlockListOfPolesToBlockTridiagonalMatrix` (`:668`) —
  per-pole Hermitian principal square roots
  (`CompactMatrixSqrt`,
  `Core/BasicMath/BasicMath_CompactMatrixVector.cpp:3316`)
  form the starting block $V_0 = [B_1\, B_2 \cdots B_M]$;
  $H_\mathrm{LP} = \mathrm{diag}(a_i I_N)$;
  then `DiagonalMatrixBlockTridiagonalize` (`:3999`)
  → `CompactMatrixBlockBandDiagonalize`
  (`BasicMath_CompactMatrixVector.cpp:4985` real, `:5108` complex).
- Corner assembly:
  `SetTMTFromValandConInBlockNaturalImpurityOrbitalMatrixType`
  (`BasicMath_BlockNaturalImpurityOrbital.cpp`):
  builds $T$ from the first couplings $(b_{v0}, b_{c0})$ as
  $T = ((b_{v0}, b_{c0}), (b_{c0}, -b_{v0})) \oplus I$
  and stores $TMT = T \cdot M \cdot T^C$ —
  the block version of the draft's $U_N H'_N U_N$.
- Evaluation and conversion:
  `EvaluateBlockNaturalImpurityOrbital` (`BlockTypeConversion.cpp:308`),
  `BlockNaturalImpurityOrbitalMatrixToSpectra(+WithBroadening)`,
  `...ToCompactMatrix(+WithDeflation)`,
  chopping via `BlockNaturalImpurityOrbitalMatrixChop(ReduceDimension)`.

### Numerical stabilization of the block Lanczos

`CompactMatrixBlockBandDiagonalize` orthogonalizes each new block with
**SVD** (`CompactMatrixSVDOrthogonalizeRow`, LAPACK `dgesvd`/`zgesvd`),
using the `SingularValue` threshold as rank-revealing **deflation**
(directions with small singular values are dropped),
plus optional full Gram-Schmidt reorthogonalization against all
previous blocks (`ReOrthogonalize` flag).
This handles degenerate subspaces / multi-dimensional irreps.
Answers the `\mwh` note in the draft, which suggests QR:
the implementation uses SVD with deflation, not QR.

## Worked Example: Five-Pole Particle-Hole-Symmetric Bath

Scalar (single-band) case with all steps analytic;
an odd, symmetric pole set forces a pole exactly at $E_F$,
so it also exercises the half/half sharing rule.
Input: normalized hybridization with five equal-weight poles,

$$
\Delta(\omega) = \sum_{k=1}^{5} \frac{1/5}{\omega - a_k + \mathrm{i}0^+},
\qquad a = (-2, -1, 0, 1, 2), \qquad \mu = 0 .
$$

Star geometry: impurity at $\epsilon_d = 0$ plus five bath levels
$\epsilon_k = a_k$ with couplings $V_k = 1/\sqrt{5}$.

### Split

The pole at $\omega = 0$ contributes half to each part
(`ListOfPolesCopyVal/Con` window rule; the Fermi function gives the
same $f(0) = 1/2$ at any temperature):

| part       | poles     | weights        | $B_0$        |
| ---------- | --------- | -------------- | ------------ |
| valence    | $-2,-1,0$ | $1/5,1/5,1/10$ | $1/\sqrt{2}$ |
| conduction | $0,1,2$   | $1/10,1/5,1/5$ | $1/\sqrt{2}$ |

Normalization $(B_0^v)^2 + (B_0^c)^2 = 1$ holds.

### Chains (exact Lanczos coefficients)

Valence part: Lanczos on $\mathrm{diag}(-2,-1,0)$ with normalized
start vector $(\sqrt{2/5}, \sqrt{2/5}, \sqrt{1/5})$ gives

$$
a_v = \left(-\tfrac{6}{5},\, -\tfrac{33}{35},\, -\tfrac{6}{7}\right),
\qquad
b_v = \left(\tfrac{\sqrt{14}}{5},\, \tfrac{2\sqrt{5}}{7}\right)
\approx (0.748331,\, 0.638877) .
$$

Trace check: $-\tfrac{6}{5} - \tfrac{33}{35} - \tfrac{6}{7} = -3$
= sum of the valence pole energies.
Conduction part by particle-hole symmetry:
$a_c = +\left(\tfrac{6}{5}, \tfrac{33}{35}, \tfrac{6}{7}\right)$,
same $b_c = b_v$.

### Corner (`SetBADoubleTriDiagonalMatrixConVal`)

With $b_{v0} = b_{c0} = 1/\sqrt{2}$ after normalization
($b_0 = 1$ since $\Delta$ is normalized):

$$
a_{cvB} = a_{cvA} = 0, \qquad
a_{cvBA} = \tfrac{1}{2}\left(-\tfrac{6}{5} - \tfrac{6}{5}\right)
= -\tfrac{6}{5},
$$

$$
b_{vB0} = b_{cB0} = b_{vA0} = \tfrac{\sqrt{7}}{5} \approx 0.529150,
\qquad
b_{cA0} = -\tfrac{\sqrt{7}}{5} .
$$

At particle-hole symmetry the bonding/antibonding sites sit at zero
energy and all information about $a_{v1} = -a_{c1}$ moves into their
coupling $a_{cvBA}$.

### Assembled single-particle Hamiltonian

Basis order $(i, B, A, v_2, v_3, c_2, c_3)$,
with $t = \sqrt{7}/5$ and $b_2 = 2\sqrt{5}/7$:

$$
H_N =
\begin{pmatrix}
0 & 1 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & -\tfrac{6}{5} & t & 0 & t & 0 \\
0 & -\tfrac{6}{5} & 0 & t & 0 & -t & 0 \\
0 & t & t & -\tfrac{33}{35} & b_2 & 0 & 0 \\
0 & 0 & 0 & b_2 & -\tfrac{6}{7} & 0 & 0 \\
0 & t & -t & 0 & 0 & \tfrac{33}{35} & b_2 \\
0 & 0 & 0 & 0 & 0 & b_2 & \tfrac{6}{7}
\end{pmatrix} .
$$

The impurity couples to a single bath orbital $B$ (strength $b_0 = 1$);
$B$ is the analog of the papers' fractional orbital $b$,
and the sign $-t$ on the $A$-conduction bond is the interference
cancellation.

### Verification (numerical diagonalization)

- One-particle spectrum of the star geometry:
  $\pm 2.115668$, $\pm 1.181487$, $\pm 0.357823$.
  $H_N$ reproduces it plus one extra eigenvalue at exactly $0$
  carrying **zero impurity weight**:
  the $E_F$ pole donated a site to _each_ chain
  (6 bath orbitals instead of 5),
  and only the symmetric combination of the two half-weight modes
  couples back. The redundancy is harmless.
- Impurity spectral weights match the star values
  $(0.061780, 0.132001, 0.306219)$ per mirror pair,
  and $\Delta(z)$ from $H_N$ equals the pole sum at arbitrary complex
  $z$ to machine precision.
- Occupations of the $U = 0$ ground state
  (zero mode half-filled, per spin):

  | site | $i$ | $B$ | $A$ | $v_2$    | $v_3$    | $c_2$    | $c_3$    |
  | ---- | --- | --- | --- | -------- | -------- | -------- | -------- |
  | $n$  | 1/2 | 1/2 | 1/2 | 0.772493 | 0.901670 | 0.227507 | 0.098330 |

  Particle-hole symmetry gives $n_{v_l} + n_{c_l} = 1$ exactly.
  The chains approach 0/1 monotonically but are **not** exactly 0/1:
  the Green's-function-level split uses the _decoupled_ bath as
  reference (Fermi function of the pole energies),
  and this example is maximally unfavorable —
  a pole at $E_F$ and strong total hybridization $b_0 = 1$.
  With the papers' coupled-reference density matrix (2014 App. E)
  the $U = 0$ occupations outside the $(i, b)$ pair would be exactly
  0 and 1; the pole-energy split trades that exactness for a
  construction that never requires solving the coupled one-particle
  problem, and the residual leakage is what the adaptive determinant
  selection absorbs.

## Notes for the DiscreteDMFT Draft

- App. A.4 currently only describes the Fermi-function split.
  The code additionally implements
  (a) the T = 0 linear $\varepsilon$-window split and
  (b) the occupation-constrained split
  (`ListOfPolesToBADoubleTridiagonalMatrixOccupation`);
  the latter makes the DMFT loop robust when the impurity filling is
  not at a pole boundary and connects to the 2014 reference system.
- The proposed block-Lanczos appendix can describe the SVD-with-deflation
  scheme actually implemented (see above) instead of plain QR.
- Hermitian $B_i$ blocks: the code enforces them via principal square
  roots per pole; after Lanczos the off-diagonal blocks can be made
  Hermitian by unitary rotations within each block
  (as stated in the draft, App. A.2).

## Not Covered Here

- The DMRG side of the 2019 paper (matrix-product-state solver with
  restricted active space) is not part of this Quanty repository;
  the repo implements the ED/Krylov path.
- The Lua-facing API (`LuaInterface/ResponseFunction.cpp`, `ssp.cpp`)
  exposes the formats but was not traced in detail.
- The 2014 iterative impurity-block rotation
  (density matrix of the impurity orbitals re-diagonalized during the
  ground-state iteration, 2014 App. E) — the scalar single-band DMFT
  code does not need it; where the multiorbital solver applies it was
  not traced.
