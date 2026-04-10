import json

from fiatlux.core.grid import Grid
from fiatlux.core.source import PlaneWave
from fiatlux.optics.elements.mask import *
from fiatlux.optics.elements.deformable_mirror import *

from fiatlux.config.builder import (
    build_serial_elements,
)

from fiatlux.core.spectrum import (
    Spectrum,
    PhotometricBand,
)

from fiatlux.optics.detector import Detector
from fiatlux.system.optical_system import SerialSystem


def from_json(path):

    with open(path) as f:
        config = json.load(f)

    objects = {}

    # ------------------
    # Spectrum
    # ------------------

    spec_cfg = config["source"]["spectrum"]

    band = PhotometricBand[spec_cfg["band"]]

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

    detector = Detector(grid=pupil_grid)

    # ------------------
    # System
    # ------------------

    system = SerialSystem(source=source, detector=detector, elements=serial_elements)

    return system
