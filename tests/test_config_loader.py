import json
from pathlib import Path

import pytest
import torch

from fiatlux.config.loader import SimulationSetup, from_json
from fiatlux.core.grid import Grid
from fiatlux.core.source import PlaneWave
from fiatlux.core.spectrum import PhotometricBand, Spectrum
from fiatlux.optics.detector import Detector
from fiatlux.optics.elements.mask import CircularAperture
from fiatlux.optics.propagator import MFTPropagator
from fiatlux.system.optical_system import SerialSystem


CONFIG_PATH = Path(__file__).parents[1] / "config" / "minimal.json"


def build_equivalent_python_setup():
    pupil_grid = Grid(nx=8, ny=6, dx=0.1, dy=0.12)
    focal_grid = Grid(nx=10, ny=4, dx=1e-6, dy=1.5e-6)
    source = PlaneWave(Spectrum(0, PhotometricBand.HCM2, samples=2))
    elements = [
        CircularAperture(pupil_grid, radius=0.31),
        MFTPropagator(focal_length=2.0, output_grid=focal_grid),
    ]
    detector = Detector(focal_grid, exposure_time=1.0, quantum_efficiency=1.0)
    return SimulationSetup(SerialSystem(elements), source, detector, {})


def test_minimal_json_builds_current_serial_system_api():
    setup = from_json(CONFIG_PATH)

    assert isinstance(setup, SimulationSetup)
    assert isinstance(setup.system, SerialSystem)
    assert [type(element) for element in setup.system.elements] == [
        CircularAperture,
        MFTPropagator,
    ]
    assert setup.detector.grid is setup.objects["focal_grid"]


def test_json_and_python_systems_produce_equivalent_results():
    configured = from_json(CONFIG_PATH)
    python = build_equivalent_python_setup()

    configured_result = configured.run()
    python_result = python.run()

    torch.testing.assert_close(
        configured.detector.image_buffer,
        python.detector.image_buffer,
    )
    torch.testing.assert_close(
        configured_result.steps[-1].field_after.complex_amplitude,
        python_result.steps[-1].field_after.complex_amplitude,
    )


def test_unknown_detector_grid_has_actionable_error(tmp_path):
    config = json.loads(CONFIG_PATH.read_text())
    config["detector"]["grid"] = "missing_grid"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="Unknown detector grid reference"):
        from_json(path)
