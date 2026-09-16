"""Import a sampled OPD power spectrum without resampling or RMS fitting."""
import math

import torch

from .atmosphere import AtmosphereModel


class TabulatedAtmosphereModel(AtmosphereModel):
    """Independent Gaussian OPD screens from centered power per Fourier bin.

    ``power_nm2`` is a real ``(ny, nx)`` array in nm² per bin, as returned
    by TIPTOP ``simulation.PSD[source]`` (not a density in nm² m²).
    ``frequency_step`` is a positive scalar in cycles/metre. The conjugate
    spatial grid must be supplied by FIATLUX and is never changed.

    Real screens require inversion symmetry. Asymmetric input is rejected
    unless ``symmetrize=True`` explicitly averages opposite frequency bins,
    conserving total power. No RMS normalization or second piston filter is
    applied. Sampling removes only the DC bin by default, like AtmosphereModel.
    A spatial PSD alone does not specify temporal correlations.
    """

    def __init__(self, grid, power_nm2, *, frequency_step, reference_wavelength=500e-9,
                 symmetrize=False, seed=None, generator=None):
        power = torch.as_tensor(power_nm2, device=grid.device)
        if power.is_complex() or power.ndim != 2 or min(power.shape) < 2:
            raise ValueError("power_nm2 must be a real 2D array with each size >= 2.")
        if power.dtype not in (torch.float32, torch.float64):
            power = power.to(torch.float64)
        if not torch.isfinite(power).all() or (power < 0).any():
            raise ValueError("power_nm2 must be finite and nonnegative.")
        step = float(frequency_step)
        if not math.isfinite(step) or step <= 0:
            raise ValueError("frequency_step must be positive and finite.")
        if not math.isfinite(reference_wavelength) or reference_wavelength <= 0:
            raise ValueError("reference_wavelength must be positive and finite.")
        if tuple(power.shape) != grid.shape:
            raise ValueError("PSD shape must match the existing FIATLUX grid.")
        for size, pitch in [(grid.nx, grid.dx), (grid.ny, grid.dy)]:
            if not math.isclose(step, 1 / (size * pitch), rel_tol=1e-10, abs_tol=1e-14):
                raise ValueError("PSD frequency increment does not match the FIATLUX grid.")
        power = power.to(dtype=grid.dtype)
        native = torch.fft.ifftshift(power).clone()
        opposite = torch.roll(torch.flip(native, (-2, -1)), (1, 1), (-2, -1))
        total = native.sum()
        self.inversion_asymmetry = float((native - opposite).abs().sum() / total) if total > 0 else 0.0
        if not symmetrize and not torch.allclose(native, opposite, rtol=1e-6, atol=0):
            raise ValueError("Power lacks inversion symmetry; set symmetrize=True explicitly.")
        self.power_nm2 = (native + opposite) / 2
        self._frequency_step = step
        super().__init__(grid, reference_wavelength=reference_wavelength,
                         seed=seed, generator=generator)

    def compute_phase_psd(self):
        return self.power_nm2 * (2 * math.pi * 1e-9 / self.reference_wavelength) ** 2 / self._frequency_step ** 2
