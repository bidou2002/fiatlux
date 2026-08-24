"""
fiatlux — Optical propagation and simulation framework.

High-level API exposing the main simulation objects.

Typical usage:
    from fiatlux import Grid, Field, Source, Spectrum
    from fiatlux import OpticalSystem
"""

# Version (optional but recommended)
__version__ = "0.1.0"

# =========================
# Core physics objects
# =========================

from .core.grid import Grid
from .core.field import Field
from .core.source import Source, PlaneWave, GaussianSource
from .core.spectrum import Spectrum, PhotometricBand

# =========================
# Optical system
# =========================

from .system.optical_system import SerialSystem, SimulationResult
from .system.interaction_matrix import InteractionMatrix

# =========================
# Optics (propagation etc.)
# =========================

from .optics.propagator import Propagator, IdentityPropagator, MFTPropagator
from .optics.detector import Detector
from .optics.adc import ADCDispersionModel
from .optics.atmosphere import AtmosphereModel, KolmogorovAtmosphereModel, NCPAModel

# =========================
# Optical elements
# =========================

from .optics.elements.base import OpticalElement
from .optics.elements.deformable_mirror import (
    DeformableMirror,
    ActuatorGrid,
    GaussianZonalBasis,
    FourierBasis,
    SquareZonalBasis,
    SquarePTTZonalBasis,
    ZernikeBasis,
)
from .optics.elements.field_stop import ShanonFieldStop
from .optics.elements.mask import (
    Mask,
    CircularAperture,
    ArbitraryAperture,
    ZeldaMask,
    ZeldaStop,
    ADC,
    Atmosphere,
    NCPA,
    Random,
    TipTilt,
    Piston,
    HarmoniResiduals,
)

# =========================
# Config system
# =========================

from .config.loader import from_json
from .config.builder import build_serial_elements

# =========================
# Utilities
# =========================

from .utils.converter import *
from .utils.fits_loader import *
from .utils.resolution import *
from .utils.zernike import *

# =========================
# Public API control
# =========================

__all__ = [
    # Core
    "Grid",
    "Field",
    "Source",
    "PlaneWave",
    "GaussianSource",
    "Spectrum",
    "PhotometricBand",
    # System
    "SerialSystem",
    "SimulationResult",
    "InteractionMatrix",
    # Optics
    "Propagator",
    "IdentityPropagator",
    "MFTPropagator",
    "Detector",
    "ADCDispersionModel",
    "AtmosphereModel",
    "KolmogorovAtmosphereModel",
    "NCPAModel",
    # Elements
    "OpticalElement",
    "DeformableMirror",
    "ActuatorGrid",
    "GaussianZonalBasis",
    "FourierBasis",
    "SquareZonalBasis",
    "SquarePTTZonalBasis",
    "ZernikeBasis",
    "ShanonFieldStop",
    "Mask",
    "CircularAperture",
    "ArbitraryAperture",
    "ZeldaMask",
    "ZeldaStop",
    "ADC",
    "Atmosphere",
    "NCPA",
    "Random",
    "TipTilt",
    "Piston",
    "HarmoniResiduals",
    # Config
    "from_json",
    "build_serial_elements",
]
