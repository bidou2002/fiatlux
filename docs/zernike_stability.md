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
`PupilOrthogonalizedBasis` now computes the symmetric Löwdin transform from a
direct SVD of `sqrt(W) @ B`, restricted to positive-weight pixels. It does not
form `G = B.T @ W @ B` in the construction path.

Existing calls retain the Gram-relative meaning of `rcond`, including the
default `eps(input_dtype) * n_modes`. The equivalent singular cutoff is its
square root. Explicitly selecting `rcond_on="singular_values"` uses a relative
singular-value cutoff instead, with default
`eps(input_dtype) * max(n_positive_pixels, n_modes)`.

Non-finite weighted matrices, SVD results or output modes, and SVD convergence
failures, raise explicit numerical errors before a usable basis is returned.
A rank failure reports the numerical rank, singular cutoff and spectral tail;
no directions are silently removed. `diagnose(compare_gram=True)` also returns
the right singular combinations and compares with the Gram at the equivalent
cutoff. See [the full SVD/ELT investigation](pupil_orthogonalization.md).

Reproduce the complete high-order case from an installed checkout:

```bash
python benchmarks/zernike_stability.py
```

This legacy-policy benchmark retains all 3240 requested modes on a 128 × 128 grid, in both
precisions, and uses the analytical ELT pupil with its default obscuration and
spiders (40 m sampled extent). It compares the Gram-based rank with an
independent float64 SVD of `sqrt(W) @ B`, applying the **same Gram cutoff** to
the squared singular values. When orthogonalization succeeds, it reports the
maximum entry and relative Frobenius norm of `B.T @ W @ B - I`.

For this particular 40 m high-order sampled ELT case, the independent SVD
confirms directions below even the singular-value default cutoff. This is
distinct from the former overflow and false rank-zero diagnostic. The tests
also cover a fully retained 3240-mode float64 case at 37 m, including weighted
orthogonality, reconstruction and whole-grid subspace preservation. The 37 m
and 40 m domains are different inputs and must not be compared as if only the
solver had changed.
