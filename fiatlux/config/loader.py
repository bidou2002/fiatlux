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
from fiatlux.config.schema import ConfigurationError, validate_config

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

    try:
        with open(path) as f:
            config = json.load(f)
    except json.JSONDecodeError as error:
        raise ConfigurationError(
            f"Invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}."
        ) from error

    validate_config(config)

    objects = {}

    # ------------------
    # Spectrum
    # ------------------

    spec_cfg = config["source"]["spectrum"]

    try:
        band = PhotometricBand[spec_cfg["band"]]
    except KeyError as error:
        raise ConfigurationError(
            f"source.spectrum.band: {error.args[0]}"
        ) from error

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

    try:
        serial_elements = build_serial_elements(config["serial_elements"], objects)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"serial_elements: {error}") from error

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
    try:
        detector = Detector(grid=detector_grid, **detector_cfg)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"detector: {error}") from error
    objects["detector"] = detector

    # ------------------
    # System
    # ------------------

    system = SerialSystem(elements=serial_elements)
    _validate_grid_chain(system, detector)
    objects["system"] = system

    return SimulationSetup(system, source, detector, objects)


def _validate_grid_chain(system: SerialSystem, detector: Detector) -> None:
    """Reject incompatible element and detector grids before propagation."""
    current_grid = getattr(system.elements[0], "grid", None)
    if current_grid is None:
        current_grid = getattr(system.elements[0], "pixel_grid", None)
    if current_grid is None:
        raise ConfigurationError(
            "serial_elements.0 does not define the source-generation grid."
        )

    for index, element in enumerate(system.elements):
        expected_grid = getattr(element, "pixel_grid", None)
        if expected_grid is None and not hasattr(element, "output_grid"):
            expected_grid = getattr(element, "grid", None)
        if expected_grid is not None and expected_grid != current_grid:
            raise ConfigurationError(
                f"serial_elements.{index} ({type(element).__name__}) uses a grid "
                "incompatible with the preceding optical plane."
            )
        if hasattr(element, "output_grid"):
            current_grid = element.output_grid

    if detector.grid != current_grid:
        raise ConfigurationError(
            "detector.grid is incompatible with the final optical plane."
        )
