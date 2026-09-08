"""Statistical models for atmospheric and instrumental OPD screens.

Frequency coordinates are in cycles per metre. The phase power spectral
density follows the usual two-dimensional von Karman convention

    W_phi(f) = 0.023 r0**(-5/3) (f**2 + 1/L0**2)**(-11/6),

and has units radian**2 metre**2. ``r0`` is defined at
``reference_wavelength``. A Kolmogorov spectrum is obtained with
``outer_scale=None``.

Screens are synthesized directly from this PSD using the frequency-bin area;
there is no empirical RMS renormalization or spatial-frequency filter. The
only exceptional Fourier coefficient is piston: it is removed by default
because it is unobservable in a single pupil, and because the pure Kolmogorov
spectrum diverges at zero frequency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math

import torch

from fiatlux.core.grid import Grid


class AtmosphereModel(ABC):
    """Base class for phase-screen statistics on a fixed spatial grid."""

    def __init__(
        self,
        grid: Grid,
        *,
        reference_wavelength: float,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if grid.nx < 2 or grid.ny < 2:
            raise ValueError("Atmospheric grids require nx >= 2 and ny >= 2.")
        if grid.dx <= 0 or grid.dy <= 0:
            raise ValueError("Atmospheric grid spacings dx and dy must be positive.")
        if reference_wavelength <= 0:
            raise ValueError("reference_wavelength must be positive.")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("dtype must be torch.float32 or torch.float64.")

        self.grid = grid
        self.reference_wavelength = float(reference_wavelength)
        self.dtype = dtype
        self.generator = torch.Generator(device=grid.device)
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)

        # PSD arrays use native, unshifted torch.fft ordering.
        self.phase_psd = self.compute_phase_psd()

    @property
    def psd(self) -> torch.Tensor:
        """Phase PSD in radian^2 metre^2, in native FFT ordering."""
        return self.phase_psd

    @property
    def frequency_bin_area(self) -> float:
        """Area df_x df_y of one discrete Fourier-frequency bin."""
        dfx = 1.0 / (self.grid.nx * self.grid.dx)
        dfy = 1.0 / (self.grid.ny * self.grid.dy)
        return dfx * dfy

    def frequency_grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(fx, fy)`` arrays shaped ``(ny, nx)`` in cycles/m."""
        fx = torch.fft.fftfreq(
            self.grid.nx,
            d=self.grid.dx,
            device=self.grid.device,
            dtype=self.dtype,
        )
        fy = torch.fft.fftfreq(
            self.grid.ny,
            d=self.grid.dy,
            device=self.grid.device,
            dtype=self.dtype,
        )
        fy, fx = torch.meshgrid(fy, fx, indexing="ij")
        return fx, fy

    @abstractmethod
    def compute_phase_psd(self) -> torch.Tensor:
        """Return the two-dimensional phase PSD in radian^2 metre^2."""

    def sample_phase(
        self,
        *,
        generator: torch.Generator | None = None,
        remove_piston: bool = True,
    ) -> torch.Tensor:
        """Draw a real phase screen in radians at the reference wavelength.

        Filtering unit-variance real white noise automatically provides the
        Hermitian Fourier coefficients required for a real-valued screen. With
        PyTorch's FFT normalization, the transfer function is
        ``sqrt(N * W_phi * df_x * df_y)``, where ``N = nx * ny``.
        """
        rng = self.generator if generator is None else generator
        white = torch.randn(
            (self.grid.ny, self.grid.nx),
            dtype=self.dtype,
            device=self.grid.device,
            generator=rng,
        )

        transfer = torch.sqrt(
            self.phase_psd
            * self.frequency_bin_area
            * (self.grid.nx * self.grid.ny)
        )
        if remove_piston:
            transfer = transfer.clone()
            transfer[0, 0] = 0.0

        return torch.fft.ifft2(torch.fft.fft2(white) * transfer).real

    def sample_opd(
        self,
        *,
        generator: torch.Generator | None = None,
        remove_piston: bool = True,
    ) -> torch.Tensor:
        """Draw an optical-path-difference screen in metres."""
        phase = self.sample_phase(
            generator=generator,
            remove_piston=remove_piston,
        )
        return phase * (self.reference_wavelength / (2.0 * math.pi))


class KolmogorovAtmosphereModel(AtmosphereModel):
    """Kolmogorov or von Karman phase-screen model.

    Parameters
    ----------
    grid
        Spatial sampling grid in metres.
    r0
        Fried parameter in metres at ``reference_wavelength``.
    reference_wavelength
        Wavelength in metres at which ``r0`` and phase are defined.
    outer_scale
        Von Karman outer scale ``L0`` in metres. ``None`` selects the pure
        Kolmogorov spectrum. No ad-hoc high-pass correction is applied.
    seed
        Optional seed for this model's private random-number generator.
    dtype
        Real floating-point dtype used for the PSD and generated screens.
    """

    def __init__(
        self,
        grid: Grid,
        r0: float,
        reference_wavelength: float = 500e-9,
        *,
        outer_scale: float | None = None,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if r0 <= 0:
            raise ValueError("r0 must be positive.")
        if outer_scale is not None and outer_scale <= 0:
            raise ValueError("outer_scale must be positive or None.")

        self.r0 = float(r0)
        self.outer_scale = None if outer_scale is None else float(outer_scale)
        super().__init__(
            grid,
            reference_wavelength=reference_wavelength,
            seed=seed,
            dtype=dtype,
        )

    def compute_phase_psd(self) -> torch.Tensor:
        fx, fy = self.frequency_grid()
        frequency_squared = fx.square() + fy.square()

        if self.outer_scale is None:
            radial_term = frequency_squared.pow(-11.0 / 6.0)
            # Pure Kolmogorov piston is divergent and physically unobservable.
            radial_term[0, 0] = 0.0
        else:
            radial_term = (frequency_squared + 1.0 / self.outer_scale**2).pow(
                -11.0 / 6.0
            )

        return 0.023 * self.r0 ** (-5.0 / 3.0) * radial_term


class NCPAModel:
    """Stationary isotropic power-law model for instrumental NCPA.

    Unlike atmospheric phase statistics, NCPA are defined directly as an
    optical-path-difference field. Frequencies are in cycles per metre and
    ``opd_psd`` has units metre**4 (OPD**2 per inverse-square-metre frequency
    area). Its integral over the sampled two-dimensional frequency plane is
    ``opd_rms**2``.

    The requested RMS normalizes the *ensemble PSD*, not each random draw.
    Individual realizations therefore have naturally fluctuating RMS; no
    post-generation rescaling or hidden PSD shaping is performed.

    Parameters
    ----------
    grid
        Spatial sampling grid in metres.
    opd_rms
        Ensemble RMS optical path difference in metres, excluding piston.
    spectral_index
        Positive exponent ``alpha`` of the radial power law ``f**(-alpha)``.
    outer_scale
        Optional low-frequency roll-off scale in metres. If supplied, the
        shape is ``(f**2 + 1/L0**2)**(-alpha/2)``. Piston remains excluded.
    seed
        Optional seed for this model's private random-number generator.
    dtype
        Real floating-point dtype used for the PSD and generated screens.
    """

    def __init__(
        self,
        grid: Grid,
        opd_rms: float,
        *,
        spectral_index: float = 3.0,
        outer_scale: float | None = None,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if grid.nx < 2 or grid.ny < 2:
            raise ValueError("NCPA grids require nx >= 2 and ny >= 2.")
        if grid.dx <= 0 or grid.dy <= 0:
            raise ValueError("NCPA grid spacings dx and dy must be positive.")
        if opd_rms <= 0:
            raise ValueError("opd_rms must be positive.")
        if spectral_index <= 0:
            raise ValueError("spectral_index must be positive.")
        if outer_scale is not None and outer_scale <= 0:
            raise ValueError("outer_scale must be positive or None.")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("dtype must be torch.float32 or torch.float64.")

        self.grid = grid
        self.opd_rms = float(opd_rms)
        self.spectral_index = float(spectral_index)
        self.outer_scale = None if outer_scale is None else float(outer_scale)
        self.dtype = dtype
        self.generator = torch.Generator(device=grid.device)
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)

        self.opd_psd = self.compute_opd_psd()

    @property
    def psd(self) -> torch.Tensor:
        """OPD PSD in metre**4, in native unshifted FFT ordering."""
        return self.opd_psd

    @property
    def frequency_bin_area(self) -> float:
        """Area ``df_x df_y`` of one Fourier-frequency bin in metre**-2."""
        return 1.0 / (
            self.grid.nx
            * self.grid.dx
            * self.grid.ny
            * self.grid.dy
        )

    def frequency_grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(fx, fy)`` arrays shaped ``(ny, nx)`` in cycles/m."""
        fx = torch.fft.fftfreq(
            self.grid.nx,
            d=self.grid.dx,
            device=self.grid.device,
            dtype=self.dtype,
        )
        fy = torch.fft.fftfreq(
            self.grid.ny,
            d=self.grid.dy,
            device=self.grid.device,
            dtype=self.dtype,
        )
        fy, fx = torch.meshgrid(fy, fx, indexing="ij")
        return fx, fy

    def compute_opd_psd(self) -> torch.Tensor:
        """Build an OPD PSD whose discrete integral is ``opd_rms**2``."""
        fx, fy = self.frequency_grid()
        frequency_squared = fx.square() + fy.square()
        if self.outer_scale is None:
            shape = frequency_squared.pow(-self.spectral_index / 2.0)
        else:
            shape = (
                frequency_squared + 1.0 / self.outer_scale**2
            ).pow(-self.spectral_index / 2.0)

        shape[0, 0] = 0.0
        normalization = self.opd_rms**2 / (
            shape.sum() * self.frequency_bin_area
        )
        return shape * normalization

    def sample_opd(
        self,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw a real, piston-free OPD screen in metres."""
        rng = self.generator if generator is None else generator
        white = torch.randn(
            (self.grid.ny, self.grid.nx),
            dtype=self.dtype,
            device=self.grid.device,
            generator=rng,
        )
        transfer = torch.sqrt(
            self.opd_psd
            * self.frequency_bin_area
            * (self.grid.nx * self.grid.ny)
        )
        return torch.fft.ifft2(torch.fft.fft2(white) * transfer).real
