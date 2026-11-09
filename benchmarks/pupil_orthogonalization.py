"""Diagnose all 3240 ELT Zernike directions without changing the rank cutoff.

Examples (CPU, float64)::

    python benchmarks/pupil_orthogonalization.py --sizes 128 256 --extent 37
    python benchmarks/pupil_orthogonalization.py --sizes 512 --extent 37
    python benchmarks/pupil_orthogonalization.py --sizes 128 --extent 40

For 512 and larger grids, bounded-memory QR blocks followed by SVD(R) replace
the dense SVD(A). This preserves A's singular spectrum without forming A.T A.
It reports the spectrum and weak combinations, but does NOT claim to have
constructed or measured the full orthogonalized 512-grid command matrix.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

from fiatlux import ELTHarmoniPupil, Grid, PupilOrthogonalizedBasis, ZernikeBasis
from fiatlux.core.spectrum import Band, Spectrum
from fiatlux.utils.zernike import noll_indices, zernike_basis


def streamed_qr(blocks, n_modes, *, progress=False):
    """Return R and an independently accumulated Gram for a stream of A rows."""
    triangular = None
    gram = None
    for index, block in enumerate(blocks):
        if not block.numel():
            continue
        contribution = block.mT @ block
        gram = contribution if gram is None else gram + contribution
        stacked = block if triangular is None else torch.cat((triangular, block))
        triangular = torch.linalg.qr(stacked, mode="r").R
        if progress:
            print(f"weighted QR block {index + 1}", file=sys.stderr, flush=True)
    if triangular is None or triangular.shape[0] < n_modes:
        raise ValueError("Not enough positive-weight rows for the requested benchmark.")
    return triangular, gram


def rows(basis, start, stop, rho, theta):
    return zernike_basis(basis.n_modes + 1, rho=rho[start:stop],
                         theta=theta[start:stop])[1:].mT


def regions(grid, pupil, rho, weights):
    disk = rho <= 1
    x, y = grid.meshgrid()
    radius = torch.hypot(x, y).flatten()
    central = disk & (radius < pupil.central_obscuration_radius)
    spiders = torch.zeros_like(disk)
    for angle in pupil.spider_angles:
        angle = math.radians(angle)
        distance = (-math.sin(angle)*x + math.cos(angle)*y).abs().flatten()
        spiders |= distance <= pupil.spider_width/2
    spiders &= disk & ~central & (weights == 0)
    other = disk & (weights == 0) & ~central & ~spiders
    return dict(illuminated=disk & (weights > 0), central_obscuration=central,
                spiders=spiders, other_occulted=other)


def diagnose(size, extent, count, block_rows, force_streamed, circular_mask):
    start = time.perf_counter()
    grid = Grid(size, size, extent/size, extent/size, dtype=torch.float64)
    pupil = ELTHarmoniPupil(grid, circular_mask=circular_mask)
    pupil.build(Spectrum(0, Band(1e-6, 0., 1.), 1, dtype=grid.dtype))
    raw = ZernikeBasis(grid, count)
    policy = PupilOrthogonalizedBasis(raw, pupil, rcond_on="singular_values")
    weights = pupil.transmission.square().flatten()
    weights /= weights.sum()
    rho, theta = (coordinate.flatten() for coordinate in raw._polar_coordinates())
    positive = weights > 0
    streamed = force_streamed or size >= 512
    print(f"{size}x{size}, extent {extent} m, {count} modes: "
          f"{'QR/SVD' if streamed else 'direct SVD'}", file=sys.stderr, flush=True)
    matrix = None
    if streamed:
        def blocks():
            for first in range(0, weights.numel(), block_rows):
                last = min(first + block_rows, weights.numel())
                support = positive[first:last]
                block = rows(raw, first, last, rho, theta)
                yield block[support] * weights[first:last][support, None].sqrt()

        triangular, gram = streamed_qr(blocks(), count, progress=True)
        u, singular, vh = torch.linalg.svd(triangular, full_matrices=False)
        del triangular
    else:
        matrix, _, _, u, singular, vh = policy._weighted_svd()
        gram = matrix.mT @ (weights[:, None]*matrix)
    spectrum = policy._rank_diagnostics(weights, singular, vh)
    eigenvalues = torch.linalg.eigvalsh((gram + gram.mT)/2)
    del gram
    legacy_rcond = torch.finfo(grid.dtype).eps * count
    legacy_singular_cutoff = math.sqrt(legacy_rcond)*singular[0]
    report = dict(
        size=size, extent_m=extent, dtype=str(grid.dtype), device=str(grid.device),
        modes=count, positive_pixels=int(positive.sum()),
        illuminated_pixels_outside_zernike_disk=int((positive & (rho > 1)).sum()),
        circular_mask=circular_mask, central_obscuration_radius_m=5.5,
        spider_width_m=.5, solver="streamed_QR_then_SVD_R" if streamed else "direct_SVD_A",
        sigma_max=float(singular[0]), sigma_min=float(singular[-1]),
        sigma_min_over_max=spectrum["sigma_min_over_max"],
        smallest_singular_values=singular[-20:].tolist(),
        direct_svd_relative_cutoff=spectrum["relative_cutoff"],
        direct_svd_absolute_cutoff=spectrum["singular_value_threshold"],
        direct_svd_rank=spectrum["rank"],
        legacy_gram_relative_cutoff=legacy_rcond,
        legacy_gram_absolute_cutoff=float(eigenvalues[-1]*legacy_rcond),
        legacy_gram_rank=int((eigenvalues > eigenvalues[-1]*legacy_rcond).sum()),
        svd_rank_at_legacy_gram_cutoff=int((singular > legacy_singular_cutoff).sum()),
        gram_rank_at_direct_svd_cutoff=int((eigenvalues > eigenvalues[-1]*spectrum["relative_cutoff"]**2).sum()),
        smallest_gram_eigenvalues=eigenvalues[:20].tolist(),
    )
    if not streamed:
        weighted_b = matrix[positive]*weights[positive, None].sqrt()
        sample = torch.arange(0, weighted_b.shape[0], max(1, weighted_b.shape[0]//256))
        literal = ((weighted_b[sample] @ vh.mT)/singular) @ vh
        polar_sample = u[sample] @ vh
        report["literal_vs_polar_weighted_sample_max_error"] = float((literal-polar_sample).abs().max())
        del literal, polar_sample
        # Release B before the independent full-height QR at larger samplings;
        # weak spatial combinations are evaluated later in bounded row blocks.
        if size >= 256:
            matrix = None
        # This is precisely sqrt(W) Q = U V.T, not an inverse Gram product.
        weighted_q = u @ vh
        del u
        error = weighted_q.mT @ weighted_q - torch.eye(count, dtype=grid.dtype)
        report["polar_orthogonality_error_max"] = float(error.abs().max())
        report["polar_orthogonality_error_relative_frobenius"] = float(error.norm()/math.sqrt(count))
        del error
        # Check B -> Q and Q -> B independently on the illuminated support.
        square_root = (vh.mT * singular) @ vh
        report["weighted_B_reconstruction_relative_error"] = float(
            (weighted_q @ square_root - weighted_b).norm()/weighted_b.norm())
        del square_root
        # Projection through an independent, backward-stable QR of A, with no
        # inverse singular values or rank-truncated pseudoinverse.
        q_span, _ = torch.linalg.qr(weighted_b, mode="reduced")
        del weighted_b
        report["weighted_Q_span_relative_error"] = float(
            (weighted_q-q_span @ (q_span.mT @ weighted_q)).norm()/weighted_q.norm())
        del q_span, weighted_q
        report["accepted_by_direct_svd_policy"] = spectrum["rank"] == count
    else:
        report["accepted_by_direct_svd_policy"] = spectrum["rank"] == count
        report["orthogonality_measurement"] = "not performed on the full streamed matrix"
        del u

    # Inspect the five weakest right singular combinations, including cases
    # where all pass the direct SVD policy but fail the historical Gram policy.
    indices = torch.arange(max(0, count-5), count)
    vectors = vh[indices].mT
    region_masks = regions(grid, pupil, rho, weights)
    energy = {name: torch.zeros(len(indices), dtype=grid.dtype) for name in region_masks}
    for first in range(0, weights.numel(), block_rows):
        last = min(first + block_rows, weights.numel())
        block = matrix[first:last] if matrix is not None else rows(raw, first, last, rho, theta)
        maps = (block @ vectors).square()
        for name, mask in region_masks.items():
            energy[name] += maps[mask[first:last]].sum(0)
    total = sum(energy.values())
    report["disk_area_fractions"] = {
        name: int(mask.sum())/int((rho <= 1).sum()) for name, mask in region_masks.items()
    }
    report["weak_directions"] = []
    for index, spectrum_index in enumerate(indices.tolist()):
        coefficients = vh[spectrum_index]
        largest = torch.topk(coefficients.abs(), min(8, count)).indices.tolist()
        report["weak_directions"].append(dict(
            singular_value=float(singular[spectrum_index]),
            passes_direct_svd=bool(spectrum["retained"][spectrum_index]),
            energy_fractions={name: float(values[index]/total[index]) for name, values in energy.items()},
            largest_original_mode_coefficients=[dict(
                noll=i+2, radial=noll_indices(i+2)[0], azimuthal=noll_indices(i+2)[1],
                coefficient=float(coefficients[i])) for i in largest],
        ))
    report["seconds"] = time.perf_counter() - start
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--extent", type=float, default=37.)
    parser.add_argument("--modes", type=int, default=3240)
    parser.add_argument("--block-rows", type=int, default=16384)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--streamed", action="store_true")
    parser.add_argument("--circular-mask", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    reports = []
    with torch.no_grad():
        for size in args.sizes:
            reports.append(diagnose(size, args.extent, args.modes, args.block_rows,
                                    args.streamed, args.circular_mask))
            encoded = json.dumps(dict(torch_version=torch.__version__, cases=reports), indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(encoded + "\n")
            print(json.dumps(reports[-1], indent=2), flush=True)


if __name__ == "__main__":
    main()
