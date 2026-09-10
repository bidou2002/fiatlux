"""Statistical diagnostics for spatial and temporal turbulence sequences."""

from __future__ import annotations

import math

import torch


def _real_floating_tensor(value: torch.Tensor, *, dimensions: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError("input must be a torch.Tensor.")
    if value.ndim not in dimensions:
        expected = " or ".join(str(item) for item in dimensions)
        raise ValueError(f"input must have {expected} dimensions.")
    if not value.is_floating_point():
        raise TypeError("input must have a real floating-point dtype.")


def spatial_periodogram(
    screens: torch.Tensor,
    *,
    dx: float,
    dy: float | None = None,
    remove_mean: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(fx, fy, PSD)`` with PSD units ``screen_unit² m²``.

    ``screens`` may be one ``(ny, nx)`` realization or a stack shaped
    ``(time, ny, nx)``. Multiple periodograms are averaged without retaining
    additional temporal copies.
    """
    _real_floating_tensor(screens, dimensions=(2, 3))
    dy = dx if dy is None else dy
    if not math.isfinite(dx) or dx <= 0 or not math.isfinite(dy) or dy <= 0:
        raise ValueError("dx and dy must be positive and finite metres.")

    samples = screens.unsqueeze(0) if screens.ndim == 2 else screens
    if remove_mean:
        samples = samples - samples.mean(dim=(-2, -1), keepdim=True)
    ny, nx = samples.shape[-2:]
    spectrum = torch.fft.fft2(samples, dim=(-2, -1))
    psd = (spectrum.abs().square() * (dx * dy / (nx * ny))).mean(dim=0)
    fx = torch.fft.fftfreq(nx, d=dx, device=screens.device, dtype=screens.dtype)
    fy = torch.fft.fftfreq(ny, d=dy, device=screens.device, dtype=screens.dtype)
    fy, fx = torch.meshgrid(fy, fx, indexing="ij")
    return fx, fy, psd


def spatial_structure_function(
    screens: torch.Tensor,
    *,
    spacing: float,
    max_lag: int | None = None,
    axis: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure ``D(r)=<|h(x+r)-h(x)|²>`` along x, y, or both axes."""
    _real_floating_tensor(screens, dimensions=(2, 3))
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be positive and finite metres.")
    if axis not in ("x", "y", "mean"):
        raise ValueError("axis must be 'x', 'y', or 'mean'.")
    samples = screens.unsqueeze(0) if screens.ndim == 2 else screens
    ny, nx = samples.shape[-2:]
    available = min(nx - 1, ny - 1) if axis == "mean" else (nx - 1 if axis == "x" else ny - 1)
    max_lag = min(available, 32) if max_lag is None else max_lag
    if not isinstance(max_lag, int) or isinstance(max_lag, bool) or not 1 <= max_lag <= available:
        raise ValueError(f"max_lag must be an integer between 1 and {available}.")

    values = []
    for lag in range(1, max_lag + 1):
        directions = []
        if axis in ("x", "mean"):
            directions.append((samples[..., lag:] - samples[..., :-lag]).square().mean())
        if axis in ("y", "mean"):
            directions.append((samples[..., lag:, :] - samples[..., :-lag, :]).square().mean())
        values.append(torch.stack(directions).mean())
    separations = torch.arange(
        1, max_lag + 1, device=screens.device, dtype=screens.dtype
    ) * spacing
    return separations, torch.stack(values)


def temporal_autocorrelation(
    sequence: torch.Tensor,
    *,
    max_lag: int | None = None,
    remove_mean: bool = True,
) -> torch.Tensor:
    """Return normalized temporal autocorrelation from lag zero onward."""
    _real_floating_tensor(sequence, dimensions=(3,))
    number = sequence.shape[0]
    if number < 2:
        raise ValueError("sequence must contain at least two screens.")
    max_lag = min(number - 1, 32) if max_lag is None else max_lag
    if not isinstance(max_lag, int) or isinstance(max_lag, bool) or not 0 <= max_lag < number:
        raise ValueError(f"max_lag must be an integer between 0 and {number - 1}.")
    centered = sequence - sequence.mean() if remove_mean else sequence
    variance = centered.square().mean()
    if variance <= 0:
        raise ValueError("temporal autocorrelation is undefined for zero variance.")
    correlations = [torch.ones((), device=sequence.device, dtype=sequence.dtype)]
    for lag in range(1, max_lag + 1):
        correlations.append((centered[:-lag] * centered[lag:]).mean() / variance)
    return torch.stack(correlations)


def estimate_translation(
    reference: torch.Tensor,
    shifted: torch.Tensor,
    *,
    dx: float,
    dy: float | None = None,
) -> tuple[float, float]:
    """Estimate periodic ``(shift_y, shift_x)`` in metres by cross-correlation."""
    _real_floating_tensor(reference, dimensions=(2,))
    _real_floating_tensor(shifted, dimensions=(2,))
    if reference.shape != shifted.shape:
        raise ValueError("reference and shifted screens must have the same shape.")
    dy = dx if dy is None else dy
    if not math.isfinite(dx) or dx <= 0 or not math.isfinite(dy) or dy <= 0:
        raise ValueError("dx and dy must be positive and finite metres.")

    cross = torch.fft.fft2(shifted) * torch.fft.fft2(reference).conj()
    correlation = torch.fft.ifft2(cross).real
    flat_index = int(correlation.argmax())
    ny, nx = reference.shape
    iy, ix = divmod(flat_index, nx)

    def subpixel(index: int, size: int, line: torch.Tensor) -> float:
        center = line[index]
        before = line[(index - 1) % size]
        after = line[(index + 1) % size]
        denominator = before - 2 * center + after
        correction = 0.0 if abs(float(denominator)) < 1e-30 else float(
            0.5 * (before - after) / denominator
        )
        signed = index if index <= size // 2 else index - size
        return signed + correction

    shift_y = subpixel(iy, ny, correlation[:, ix]) * dy
    shift_x = subpixel(ix, nx, correlation[iy, :]) * dx
    return shift_y, shift_x
