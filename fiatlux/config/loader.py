import json
from dataclasses import dataclass
from pathlib import Path

from fiatlux.core.grid import Grid
from fiatlux.core.source import PlaneWave
from fiatlux.optics.elements import mask as _mask_types
from fiatlux.optics.elements import deformable_mirror as _dm_types
from fiatlux.optics import propagator as _propagator_types

from fiatlux.config.builder import (
    build_serial_elements,
)

from fiatlux.core.spectrum import (
    Spectrum,
    PhotometricBand,
)

from fiatlux.optics.detector import Detector
from fiatlux.system.optical_system import SerialSystem


@dataclass
class SimulationSetup:
    """Objects required to execute a simulation loaded from configuration."""

    system: SerialSystem
    source: PlaneWave
    detector: Detector
    objects: dict[str, object]

    def run(self):
        return self.system.run(self.source, self.detector)


def from_json(path: str | Path) -> SimulationSetup:
    """Build a source, ordered optical system and detector from JSON."""

    with open(path) as f:
        config = json.load(f)

    objects = {}

    # ------------------
    # Spectrum
    # ------------------

    spec_cfg = config["source"]["spectrum"]

    band = PhotometricBand[spec_cfg["band"]]

    if "samples" in spec_cfg:
        spectrum = Spectrum(
            magnitude=spec_cfg["magnitude"],
            band=band,
            samples=spec_cfg["samples"],
        )
    else:
        spectrum = Spectrum.from_sampling(
            magnitude=spec_cfg["magnitude"], band=band, Nu=spec_cfg["Nu"]
        )

    source = PlaneWave(spectrum=spectrum)

    objects["source"] = source

    # ------------------
    # Grids
    # ------------------

    pupil_grid = Grid(**config["pupil_grid"])

    focal_grid = Grid(**config["focal_grid"])

    objects["pupil_grid"] = pupil_grid
    objects["focal_grid"] = focal_grid

    # ------------------
    # Serial elements
    # ------------------

    serial_elements = build_serial_elements(config["serial_elements"], objects)

    # ------------------
    # Detector
    # ------------------

    detector_cfg = dict(config["detector"])
    detector_grid_name = detector_cfg.pop("grid")
    try:
        detector_grid = objects[detector_grid_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown detector grid reference '{detector_grid_name}'."
        ) from error
    detector = Detector(grid=detector_grid, **detector_cfg)
    objects["detector"] = detector

    # ------------------
    # System
    # ------------------

    system = SerialSystem(elements=serial_elements)
    objects["system"] = system

    return SimulationSetup(system, source, detector, objects)
