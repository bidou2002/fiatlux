from pydantic import BaseModel, Field
from typing import List, Optional
import torch

# -------------------------------
# Define Pydantic models
# -------------------------------


class SpectrumConfig(BaseModel):
    magnitude: float
    band: str


class GridConfig(BaseModel):
    nx: int
    ny: int
    dx: float
    dy: float


class PlaneWaveConfig(BaseModel):
    spectrum: SpectrumConfig


class IdentityPropagatorConfig(BaseModel):
    grid: GridConfig


class CircularApertureConfig(BaseModel):
    radius: float


class ActuatorGridConfig(BaseModel):
    n_actuators_x: int
    n_actuators_y: int
    pitch: float


class ZonalBasisConfig(BaseModel):
    actuator_grid: ActuatorGridConfig
    pixel_grid: GridConfig
    influence_width: float


class DeformableMirrorConfig(BaseModel):
    actuator_grid: ActuatorGridConfig
    pixel_grid: GridConfig
    control_basis: ZonalBasisConfig
    stroke: float


class MFTPropagatorConfig(BaseModel):
    focal_length: float
    output_grid: GridConfig


class ZeldaMaskConfig(BaseModel):
    radius: float
    well_depth: float


class ZeldaStopConfig(BaseModel):
    radius: float


class DetectorConfig(BaseModel):
    grid: GridConfig
    photon_noise: bool = False
    readout_noise_variance: float = 0
    dark_current: float = 0
    offset: float = 0


class SerialSystemConfig(BaseModel):
    source: PlaneWaveConfig
    detector: DetectorConfig
    elements: List[str]  # names of elements in order


class InteractionMatrixConfig(BaseModel):
    dm: DeformableMirrorConfig
    poke_amplitude: float
    response_function: Optional[str] = None  # JSON cannot store lambda directly


class SystemConfig(BaseModel):
    spectrum: SpectrumConfig
    pupil_grid: GridConfig
    focal_grid: GridConfig
    source: PlaneWaveConfig
    identity_propagators: List[IdentityPropagatorConfig]
    aperture: CircularApertureConfig
    actuator_grid: ActuatorGridConfig
    basis: ZonalBasisConfig
    dm: DeformableMirrorConfig
    mfts: List[MFTPropagatorConfig]
    zelda_mask: ZeldaMaskConfig
    zelda_stop: ZeldaStopConfig
    detector: DetectorConfig
    serial_system: SerialSystemConfig
    interaction_matrix: InteractionMatrixConfig
