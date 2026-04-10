from dataclasses import dataclass
import torch

from fiatlux.core.spectrum import Spectrum
from fiatlux.optics.elements.mask import Mask


@dataclass
class ADCDispersionModel:

    def get_dispersion(self, angle: float, spectrum: Spectrum):

        theta_min = 5
        theta_max = 50
        T_mean = 7.5
        Rh_mean = 15
        P_mean = 712
        T_min = 0
        T_max = 15
        P_min = 662
        P_max = 762

        D = self._get_dispersion_unique(
            angle=angle, wl=spectrum.wavelengths, T=T_mean, Rh=Rh_mean, P=P_mean
        )

        D = (
            D
            - (
                self._get_dispersion_unique(
                    angle=theta_min,
                    wl=spectrum.wavelengths,
                    T=T_max,
                    Rh=Rh_mean,
                    P=P_min,
                )
                + self._get_dispersion_unique(
                    angle=theta_max,
                    wl=spectrum.wavelengths,
                    T=T_min,
                    Rh=Rh_mean,
                    P=P_max,
                )
            )
            / 2
        )

        return D

    def _get_dispersion_unique(
        self, angle: float, wl: torch.Tensor, T: float, Rh: float, P
    ):

        # Celsius → Kelvin
        T = T + 273.15

        PS = -10474.0 + 116.43 * T - 0.43284 * T**2 + 0.00053840 * T**3
        P2 = Rh / 100.0 * PS
        P1 = P - P2

        D1 = P1 / T * (1.0 + P1 * (57.90e-8 - (9.3250e-4 / T) + (0.25844 / T**2)))

        D2 = (
            P2
            / T
            * (
                1.0
                + P2
                * (1.0 + 3.7e-4 * P2)
                * (-2.37321e-3 + (2.23366 / T) - (710.792 / T**2) + (7.75141e4 / T**3))
            )
        )

        S0 = 1.0 / (wl.min() * 1e6)
        S = 1.0 / (wl * 1e6)

        N0_1 = 1.0e-8 * (
            (2371.34 + 683939.7 / (130 - S0**2) + 4547.3 / (38.9 - S0**2)) * D1
            + (6487.31 + 58.058 * S0**2 - 0.71150 * S0**4 + 0.08851 * S0**6) * D2
        )

        N_1 = 1.0e-8 * (
            (2371.34 + 683939.7 / (130 - S**2) + 4547.3 / (38.9 - S**2)) * D1
            + (6487.31 + 58.058 * S**2 - 0.71150 * S**4 + 0.08851 * S**6) * D2
        )

        return torch.deg2rad(
            torch.tan(torch.deg2rad(torch.as_tensor(angle)))
            * (N0_1 - N_1)
            * 206264.8
            / 3600
        )
