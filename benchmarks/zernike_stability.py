"""Reproduce high-order Zernike and ELT numerical-rank diagnostics on CPU.

Run from an installed checkout::

    python benchmarks/zernike_stability.py

The default retains all 3240 requested modes on 128 x 128 pixels. The output
distinguishes non-finite numerical failures from a finite but ill-conditioned
weighted basis. No modes are dropped and the default rcond is never changed.
"""

import argparse
import json
import time

import torch

from fiatlux import ELTHarmoniPupil, Grid, PupilOrthogonalizedBasis, ZernikeBasis
from fiatlux.core.spectrum import Band, Spectrum


def diagnose(count, size, dtype):
    start = time.perf_counter()
    grid = Grid(size, size, 40/size, 40/size, dtype=dtype)
    pupil = ELTHarmoniPupil(grid)
    pupil.build(Spectrum(0, Band(1e-6, 0., 1.), 1, dtype=dtype))
    basis = ZernikeBasis(grid, count)
    matrix = basis.build_command_matrix()
    rho, _ = basis._polar_coordinates()
    rms = matrix[rho.flatten() <= 1].square().mean(0).sqrt()
    report = dict(dtype=str(dtype), modes=count, size=size,
                  finite=bool(torch.isfinite(matrix).all()),
                  maximum=float(matrix.abs().max()),
                  disk_rms_min=float(rms.min()), disk_rms_max=float(rms.max()),
                  illuminated_pixels=int(torch.count_nonzero(pupil.transmission)))

    # Avoid rebuilding the large basis during the real production call.
    class FrozenBasis:
        pixel_grid = grid
        n_modes = count

        def build_command_matrix(self):
            return matrix

    try:
        orthogonal = PupilOrthogonalizedBasis(FrozenBasis(), pupil).build_command_matrix()
    except ValueError as error:
        report['orthogonalization'] = str(error)
    else:
        weights = pupil.transmission.double().square().flatten()
        weights /= weights.sum()
        gram_error = orthogonal.double().mT @ (weights[:, None]*orthogonal.double())
        gram_error -= torch.eye(count, dtype=torch.float64)
        report.update(orthogonalization='success',
                      gram_error_max=float(gram_error.abs().max()),
                      gram_error_relative_frobenius=float(torch.linalg.matrix_norm(gram_error)/count**.5))
        del orthogonal, gram_error

    # An independent singular-value decomposition of sqrt(W)*B checks whether
    # lost directions are genuinely unresolved by the requested Gram cutoff.
    # Use float64 even for a float32 input, without changing that input's cutoff.
    weights = pupil.transmission.double().square().flatten()
    weights /= weights.sum()
    weighted = matrix.double()[weights > 0] * weights[weights > 0, None].sqrt()
    singular_values = torch.linalg.svdvals(weighted)
    gram_threshold = torch.finfo(dtype).eps*count*singular_values[0].square()
    report.update(weighted_svd_rank_at_gram_cutoff=int((singular_values.square() > gram_threshold).sum()),
                  gram_cutoff=float(gram_threshold),
                  weighted_svd_condition=float(singular_values[0]/singular_values[-1]),
                  seconds=time.perf_counter()-start)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes', type=int, default=3240)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with torch.no_grad():
        for dtype in (torch.float32, torch.float64):
            print(json.dumps(diagnose(args.modes, args.size, dtype), indent=2), flush=True)


if __name__ == '__main__':
    main()
