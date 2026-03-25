from dataclasses import dataclass
import torch


@dataclass
class FocalPlaneConverter:
    """
    Convertisseur de coordonnées dans le plan focal.

    Paramètres physiques :
        wavelength    : longueur d'onde (m)
        pupil_diameter: diamètre de la pupille (m)
        focal_length  : longueur focale (m)
    """

    wavelength: float  # m
    pupil_diameter: float  # m
    focal_length: float  # m

    @property
    def lambda_over_D(self) -> float:
        """Taille d'une unité λ/D en mètres dans le plan focal."""
        return self.wavelength * self.focal_length / self.pupil_diameter

    @property
    def f_number(self) -> float:
        return self.focal_length / self.pupil_diameter

    # ── vers mètres ───────────────────────────────────────────────

    def lod_to_m(self, x: torch.Tensor) -> torch.Tensor:
        """λ/D → m"""
        return x * self.lambda_over_D

    def freq_to_m(self, u: torch.Tensor) -> torch.Tensor:
        """m⁻¹ → m  (x = u · λ · f)"""
        return u * self.wavelength * self.focal_length

    def rad_to_m(self, theta: torch.Tensor) -> torch.Tensor:
        """radians → m  (x = θ · f)"""
        return theta * self.focal_length

    def arcsec_to_m(self, theta: torch.Tensor) -> torch.Tensor:
        """arcsec → m"""
        return self.rad_to_m(torch.deg2rad(theta / 3600))

    # ── vers λ/D ──────────────────────────────────────────────────

    def m_to_lod(self, x: torch.Tensor) -> torch.Tensor:
        """m → λ/D"""
        return x / self.lambda_over_D

    def freq_to_lod(self, u: torch.Tensor) -> torch.Tensor:
        """m⁻¹ → λ/D"""
        return self.freq_to_m(u) / self.lambda_over_D

    # ── vers fréquences spatiales ──────────────────────────────────

    def m_to_freq(self, x: torch.Tensor) -> torch.Tensor:
        """m → m⁻¹  (u = x / (λ · f))"""
        return x / (self.wavelength * self.focal_length)

    def lod_to_freq(self, x: torch.Tensor) -> torch.Tensor:
        """λ/D → m⁻¹"""
        return self.m_to_freq(self.lod_to_m(x))

    # ── vers angles ───────────────────────────────────────────────

    def m_to_rad(self, x: torch.Tensor) -> torch.Tensor:
        """m → radians  (θ = x / f)"""
        return x / self.focal_length

    def m_to_arcsec(self, x: torch.Tensor) -> torch.Tensor:
        """m → arcsec"""
        return torch.rad2deg(self.m_to_rad(x)) * 3600

    def lod_to_arcsec(self, x: torch.Tensor) -> torch.Tensor:
        """λ/D → arcsec"""
        return self.m_to_arcsec(self.lod_to_m(x))
