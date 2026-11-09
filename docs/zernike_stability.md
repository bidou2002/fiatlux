# High-order Zernike evaluation

The radial polynomial is evaluated as

\[
R_n^{|m|}(\rho)=\rho^{|m|}
P_{(n-|m|)/2}^{(0,|m|)}(2\rho^2-1).
\]

The Jacobi three-term recurrence follows
[DLMF 18.9.1–18.9.2](https://dlmf.nist.gov/18.9.i). It avoids subtracting large
alternating terms from the explicit factorial expansion. For float32 inputs,
radial and angular calculations use float64 on the input device before casting
the result back. Autograd remains connected to the input coordinates.

`R`, `zernike`, `zernike1`, `zernike_basis`, and `ZernikeBasis` retain their
existing APIs, Noll indices, sine/cosine signs, coordinate conventions and
outside-disk values. In particular, this change does not change how
`ZernikeBasis` constructs its grid coordinates.

## Normalization and validation

Noll-normalized modes have unit mean square under the continuous unit-disk
area measure. A Cartesian 128 × 128 sampling is a different quadrature, so its
sampled RMS need not be exactly one, especially near radial order 80. Modes
are not rescaled to force that discrete RMS to unity.

Tests compare every valid radial polynomial through order 80 against the
defining polynomial evaluated with exact integer coefficients and 100-digit
decimal arithmetic. Independent Gauss–Legendre quadrature checks continuous
normalization. The 3240-mode, 128 × 128 regression checks finite values/norms,
the analytical amplitude bound, and the sampled radial-80 RMS against the same
independent reference. Other tests cover angular signs, outside masking,
dtype/device preservation, coordinate gradients and low-order regressions.

## Pupil orthogonalization

With normalized intensity weights `W = abs(P)**2 / sum(abs(P)**2)`,
`PupilOrthogonalizedBasis` uses the Gram matrix `G = B.T @ W @ B` and symmetric
Löwdin orthogonalization. Its existing default eigenvalue cutoff remains
`eps(input_dtype) * n_modes * max_eigenvalue`; an explicit `rcond` retains its
existing meaning.

Non-finite Gram matrices or eigensolver results, and eigensolver convergence
failures, raise an explicit numerical error before any rank is reported.
A finite matrix may still have a deficient **numerical** rank at the requested
precision/cutoff, from pupil obscuration, sampling or conditioning. That case
reports the rank and eigenvalue threshold. The method does not silently remove
modes or change `rcond`.

Reproduce the complete high-order case from an installed checkout:

```bash
python benchmarks/zernike_stability.py
```

The script retains all 3240 requested modes on a 128 × 128 grid, in both
precisions, and uses the analytical ELT pupil with its default obscuration and
spiders (40 m sampled extent). It compares the Gram-based rank with an
independent float64 SVD of `sqrt(W) @ B`, applying the **same Gram cutoff** to
the squared singular values. When orthogonalization succeeds, it reports the
maximum entry and relative Frobenius norm of `B.T @ W @ B - I`.

For this particular high-order sampled ELT case, finite Zernike evaluation does
not make all requested directions well-conditioned: the independent SVD can
confirm a numerical rank below 3240. This is distinct from the former overflow
and false rank-zero diagnostic. The unit tests also measure the orthogonality
error on a resolved 120-mode ELT subspace and retain a separate exact rank-loss
regression; that smaller test is not a substitute for the full 3240-mode check.
