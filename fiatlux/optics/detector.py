import torch

from fiatlux.core.grid import Grid
from fiatlux.core.field import Field
from fiatlux.optics.elements.base import validate_field_grid
from fiatlux.utils.random import RandomGeneratorMixin


class Detector(RandomGeneratorMixin):
    def __init__(
        self,
        grid: Grid,
        exposure_time: float = 0.150,
        quantum_efficiency: float = 1,
        photon_noise: bool = False,
        readout_noise_variance: float = 0,
        dark_current: float = 0,
        offset: int = 0,
        bitdepth: int = 16,
        digitize: bool = False,
        sensitivity: float = 1.0,
        random_seed: int | None = None,
        generator: torch.Generator | None = None,
        name: str = "",
    ):
        if exposure_time < 0:
            raise ValueError("exposure_time must be non-negative.")
        if not 0 <= quantum_efficiency <= 1:
            raise ValueError("quantum_efficiency must be between 0 and 1.")
        if readout_noise_variance < 0:
            raise ValueError("readout_noise_variance must be non-negative.")
        if dark_current < 0:
            raise ValueError("dark_current must be non-negative.")
        self.grid = grid
        self.exposure_time = exposure_time
        self.quantum_efficiency = quantum_efficiency
        self.photon_noise = photon_noise
        self.readout_noise_variance = readout_noise_variance
        self.dark_current = dark_current
        self.offset = offset
        if (
            not isinstance(bitdepth, int)
            or isinstance(bitdepth, bool)
            or bitdepth < 1
        ):
            raise ValueError("bitdepth must be a positive integer.")
        self.bitdepth = bitdepth
        self.digitize = digitize
        self.sensitivity = sensitivity
        self.random_seed = random_seed
        self._configure_generator(
            grid.device,
            seed=random_seed,
            generator=generator,
        )
        self.name = name
        self.image_buffer = None

    def acquire(self, field: Field) -> torch.Tensor:
        """Integrate a photon-rate-density field over pixel area and time.

        ``field.intensity()`` is interpreted as photons / s / m² in each
        wavelength channel. The returned image contains electrons, or ADUs
        when digitization is enabled.
        """
        validate_field_grid(field, self.grid, "Detector")
        photon_rate = torch.sum(field.intensity(), dim=0) * (
            self.grid.dx * self.grid.dy
        )
        photon_counts = photon_rate * self.exposure_time
        self.image_buffer = self.add_noise(
            photon_counts
        )
        return self.image_buffer

    def add_photon_noise(self, expected_electrons: torch.Tensor) -> torch.Tensor:
        """Draw detected photoelectrons from their Poisson distribution."""
        return torch.poisson(
            expected_electrons,
            generator=self.generator,
        )

    def add_dark_noise(self, electrons: torch.Tensor):
        """Add Poisson dark electrons from a rate in e-/pixel/s."""
        expected_dark_electrons = torch.full_like(
            electrons,
            self.dark_current * self.exposure_time,
        )
        dark_noise = torch.poisson(
            expected_dark_electrons,
            generator=self.generator,
        )
        return electrons + dark_noise

    def add_readout_noise(self, electrons: torch.Tensor):
        """Add zero-mean Gaussian read noise with variance in electrons²."""
        noise = torch.normal(
            mean=torch.zeros_like(electrons),
            std=self.readout_noise_variance**0.5,
            generator=self.generator,
        )
        return electrons + noise

    def photons_to_electrons(self, photons: torch.Tensor):
        return self.quantum_efficiency * photons

    def electrons_to_adus(self, electrons: torch.Tensor) -> torch.Tensor:
        """Digitize electrons into integer ADUs in the configured ADC range.

        Gain and offset are applied before values are clipped to
        ``[0, 2**bitdepth - 1]``. ``int64`` is used so a 16-bit ADC can safely
        represent its full range, including 65535.
        """
        max_adu = 2**self.bitdepth - 1
        adus = torch.floor(electrons * self.sensitivity) + self.offset
        return adus.clamp(0, max_adu).to(torch.int64)

    def add_noise(self, photon_counts: torch.Tensor) -> torch.Tensor:
        """Convert integrated photon counts through the detector pipeline."""
        electrons = self.photons_to_electrons(photons=photon_counts)
        if self.photon_noise:
            electrons = self.add_photon_noise(expected_electrons=electrons)

        # add the dark noise to electrons
        electrons = self.add_dark_noise(electrons=electrons)

        # add the computed noise to electrons
        electrons = self.add_readout_noise(electrons=electrons)

        # Convert to ADU and add baseline
        if self.digitize:
            return self.electrons_to_adus(electrons=electrons)

        return electrons
