# Weighted symmetric pupil orthogonalization

`PupilOrthogonalizedBasis` retains all requested control modes or raises a
diagnostic error. It never removes directions or adjusts the requested cutoff.
The underlying Zernike basis, its Noll ordering, normalization, orientation and
outside-disk values are unchanged.

## Construction without normal equations

Let `B` have one row per pixel and one column per original mode, and let
`w = abs(pupil.transmission)**2 / sum(abs(pupil.transmission)**2)`.
Only strictly positive weights enter the decomposition:

\[
A=\sqrt{W_+}B_+ = U\Sigma V^\mathsf{T},\qquad
T=V\Sigma^{-1}V^\mathsf{T},\qquad Q=BT.
\]

This is the same symmetric Löwdin convention as before, preserving the
original span. Forming `G = A.T @ A` first would square the condition number.
Even after a direct SVD, evaluating `B @ T` literally introduces cancellation
for very weak directions. On the illuminated support the implementation
instead evaluates the mathematically equivalent polar factor:

\[
Q_+=W_+^{-1/2}UV^\mathsf{T}.
\]

At zero-weight pixels it applies the same coefficient transform `T` to the
original basis; it does not zero out that continuation. Such values can become
very large when a mode combination is almost confined to an occulted region.
The reported orthogonality is pupil-weighted, not full-grid Euclidean
orthogonality. A small orthogonality error alone does not establish robustness
to measurement noise, pupil errors or perturbations of the original basis.

No Gram matrix is formed in `build_command_matrix()`. Non-finite weighted
matrices, SVD outputs or returned modes, and a failure of SVD convergence,
produce explicit numerical errors, rather than a misleading rank-zero report.

## Exact meaning of rcond and compatibility

The optional `rcond_on` argument distinguishes two numerical rank policies.
Both retain singular values **strictly greater than** their cutoff.

| Policy | Relative cutoff default | Retain a direction when |
| --- | --- | --- |
| `"gram"` (default, historical compatibility) | `eps(B.dtype) * n_modes` | `sigma > sqrt(rcond) * sigma_max` |
| `"singular_values"` (explicit direct-SVD policy) | `eps(B.dtype) * max(n_positive_pixels, n_modes)` | `sigma > rcond * sigma_max` |

For an explicitly supplied `rcond`, the table uses that value rather than the
default. Existing positional arguments and the historical eigenvalue-relative
meaning are preserved. The decomposition is now more accurate, so numerical
decisions at a cutoff can differ from the old rounded Gram eigenspectrum.

The two policies deliberately have different defaults. The direct-SVD default
uses the matrix-dimension-scaled machine precision convention of
[`torch.linalg.matrix_rank`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.matrix_rank.html).
Choosing it is an explicit change of rank policy, not an automatic lowering of
`rcond` in response to failure. To express the same cutoff in both domains,
use `rcond_gram = rcond_singular**2`.

```python
# Existing call: unchanged historical cutoff, evaluated through an SVD.
legacy = PupilOrthogonalizedBasis(raw_basis, pupil)

# Explicitly request the singular-value rank policy.
basis = PupilOrthogonalizedBasis(
    raw_basis, pupil, rcond_on="singular_values"
)
report = basis.diagnose(compare_gram=True)
print(report["rank"], report["sigma_min_over_max"])
print(report["singular_value_threshold"])
print(report["singular_values"][-10:])
matrix = basis.build_command_matrix()  # Recomputes with the current pupil.
```

`diagnose()` returns device-resident singular values, the retained-direction
mask and right singular vectors. Row `i` of `right_singular_vectors` contains
the original-mode coefficients of singular direction `i`, in descending
singular-value order. Rank-deficient finite bases can be inspected without
constructing an invalid orthogonal basis:

```python
coefficients = report["right_singular_vectors"][~report["retained"]]
patterns = raw_basis.build_command_matrix() @ coefficients.mT
# Even when every direction passes, inspect the five weakest combinations:
weakest_coefficients = report["right_singular_vectors"][-5:]
```

A deficient direction is generally a combination of modes, not one particular
Noll mode. The optional Gram comparison uses the **equivalent** eigenvalue
cutoff and reports Gram numerical failures separately from a successful SVD.
It does not control the construction algorithm or repair its eigenvalues.

## Reproducible 3240-mode ELT investigation

```bash
python benchmarks/pupil_orthogonalization.py --sizes 128 256 512 --extent 37 \
    --output benchmarks/results/pupil_svd_3240_extent37.json
python benchmarks/pupil_orthogonalization.py --sizes 128 --extent 40
```

The benchmark records grid size, physical extent, pupil configuration, both
rank policies, the twenty smallest singular values, Gram eigenvalues and five
weakest original-mode combinations. It also measures where those combinations
have their energy, separating central obscuration, spiders and other occulted
pixels. Energy concentration is measured rather than presumed.

Configuration: float64, CPU, `ZernikeBasis(grid, 3240)` (Noll indices 2–3241,
including the first radial-80 mode), default analytical ELT pupil with central
obscuration radius 5.5 m, projected spider width 0.5 m, segment gaps 4 mm,
`circular_mask=False`, and spacing `extent / size`. These details matter; the
grid dimensions alone do not uniquely define the numerical rank.

For 128 pixels over 37 m, PyTorch 2.15 CPU gives:

| Quantity | Measurement |
| --- | ---: |
| Positive-weight pixels | 12 110 |
| `sigma_max` | 1.9157171566 |
| `sigma_min` | 1.1542141674e-11 |
| `sigma_min / sigma_max` | 6.0249717105e-12 |
| Direct-SVD relative cutoff | 2.6889601656e-12 |
| Direct-SVD absolute cutoff | 5.1512871226e-12 |
| Direct-SVD rank | 3240 |
| Historical Gram-relative rank, by either method | 3193 |
| Gram rank at the equivalent direct-SVD cutoff | 3226 |
| `max(abs(Q.T @ W @ Q - I))` | 1.04e-14 |
| Weighted relative reconstruction error of `B` from `Q` | 1.39e-14 |
| Weighted relative projection error of `Q` onto independently QR-factored `A` | 5.73e-15 |

The 256 × 256 calculation on the same 37 m domain has `sigma_min` =
3.6029830758e-9, `sigma_min / sigma_max` = 2.1503606000e-9, and a direct-SVD
absolute cutoff of 1.7888117758e-11. It retains all 3240 directions, compared
with 3228 at the historical cutoff. Its maximum weighted orthogonality error
is 2.24e-14, weighted reconstruction error 1.54e-14 and independent weighted
span projection error 9.46e-15.

At 512 × 512 the QR/SVD spectrum has `sigma_min` = 3.9071249779e-9,
`sigma_min / sigma_max` = 2.8681073587e-9 and an absolute SVD cutoff of
5.8074772846e-11. All 3240 directions pass, versus 3228 at the historical
cutoff. The five smallest singular values are approximately
`[3.678e-8, 3.606e-8, 2.085e-8, 3.943e-9, 3.907e-9]`. The full-grid
orthogonality error was not measured in this bounded-memory run.

Complete spectra tails, cutoff comparisons and weak-combination coefficients
are recorded in the [37 m results](../benchmarks/results/pupil_svd_3240_extent37.json)
and [40 m control case](../benchmarks/results/pupil_svd_3240_extent40.json).

The unchanged historical cutoff discards 47 directions even when evaluated
accurately through singular values. That part of the old rank is a policy
decision, not simply a Gram rounding error. At the much smaller equivalent
cutoff of the direct-SVD policy, the Gram eigenspectrum itself cannot resolve
the spectrum accurately. These effects must be distinguished.

In this case the five weakest combinations put more than 99.99999996% of
their sampled disk energy inside the central obscuration. The obscuration
occupies about 9% of the sampled disk. Thus there is genuine near-dependence
on the visible pupil, although all 3240 directions pass the explicit
singular-value policy in float64. This does not prove exact algebraic rank
loss or suitability of the weakest combinations for a noisy physical system.

The 40 m normalization domain is a different basis on the physical pupil.
At 128 pixels, it has `sigma_min / sigma_max` approximately 4.27e-14: some
directions fail even the default singular-value policy (rank 3233, absolute
cutoff 4.2878237137e-12). That case is rejected
without reducing the mode count. Increasing the sampled physical extent must
not be mistaken for simply adding padding to a fixed Zernike normalization.
`ZernikeBasis` retains its existing pixel-normalized coordinates; the benchmark
also counts illuminated pixels outside its unit disk, where the original
modes remain zero. This change does not redefine that geometry.

The diagnostic benchmark also measures the unthresholded polar factor in
this rejected case. It can have a tiny orthogonality error while its directions
are below the selected rank cutoff: orthogonality alone is not a license to
accept them. The acceptance flag remains false and the production constructor
raises. Its five weakest combinations concentrate in the other occulted
regions (roughly 95–100% of disk energy), rather than the central obstruction.

For 512 and larger grids, the benchmark uses bounded-memory QR row blocks
followed by an SVD of the triangular factor. This is a stable factorization of
the weighted design matrix, without normal equations or squared singular
values in the solver. Gram accumulation is independent and only diagnostic.
The streamed path reports the spectrum and spatial combinations; it explicitly
does not report an unmeasured full-grid orthogonality error. A regression
compares its singular spectrum with a dense direct SVD on the same matrix.

## Validation and supported computation

- The independent exact-integer/100-digit polynomial reference and disk
  quadrature tests cover every valid radial polynomial through order 80.
- Low-order tests compare with the old symmetric Gram transform, verify
  continuation on occulted pixels and exercise both cutoff meanings.
- The full 3240-mode float64 ELT regression measures weighted orthogonality,
  reconstruction and whole-grid subspace preservation using an independent
  QR projection. It does not substitute a smaller basis or tune a cutoff.
- Real float32 and float64 bases retain dtype and device. Complex, apodized
  and boolean pupil transmissions use intensity weights. CPU is tested;
  CUDA cases are conditional on availability. CUDA/cuSOLVER uses the
  accuracy-oriented `gesvd` driver, following
  [PyTorch's SVD guidance](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html).
- DM command gradients and nondegenerate basis/apodization gradients are
  tested, including gradcheck. As with eigendecomposition, differentiation of
  singular vectors at repeated singular values has backend limitations; no
  gradients are silently detached in the construction path.
- The core uses PyTorch throughout, without CPU/NumPy conversions or additional
  numerical dependencies.
