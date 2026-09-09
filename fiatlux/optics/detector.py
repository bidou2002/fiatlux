import torch

from fiatlux.core.grid import Grid
from fiatlux.core.field import Field


class Detector:
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
        random_seed: int = 0,
        name: str = "",
    ):
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
        self.generator = torch.Generator().manual_seed(random_seed)
        self.name = name
        self.image_buffer = None

    def acquire(self, field: Field) -> torch.Tensor:
        self.image_buffer = self.add_noise(
            torch.sum(
                (self.grid.dx * self.grid.dy) * torch.abs(field.complex_amplitude) ** 2,
                dim=0,
            )
        )

    def add_photon_noise(self, photons):
        return torch.poisson(
            photons,
            generator=self.generator,
        )

    def add_dark_noise(self, electrons: torch.Tensor):
        dark_noise = (self.dark_current * self.exposure_time) * torch.ones(
            electrons.shape
        )
        dark_noise = torch.poisson(
            dark_noise,
            generator=self.generator,
        )
        return electrons + dark_noise

    def add_readout_noise(self, electrons: torch.Tensor):
        return (
            torch.normal(
                mean=0,
                std=self.readout_noise_variance**0.5,
                size=electrons.shape,
                generator=self.generator,
            )
            + electrons
        )

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

    def add_noise(self, photons):
        """
        CMOS noise simulation following https://arxiv.org/pdf/1412.4031.pdf
        """
        # manage photon noise
        if self.photon_noise == True:
            photons = self.add_photon_noise(photons=photons)
        else:
            photons = photons

        # convert photons to electrons
        electrons = self.photons_to_electrons(photons=photons)

        # add the dark noise to electrons
        electrons = self.add_dark_noise(electrons=electrons)

        # add the computed noise to electrons
        electrons = self.add_readout_noise(electrons=electrons)

        # Convert to ADU and add baseline
        if self.digitize:
            return self.electrons_to_adus(electrons=electrons)

        return electrons
