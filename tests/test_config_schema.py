import json
from copy import deepcopy
from pathlib import Path

import pytest

from fiatlux.config.loader import from_json
from fiatlux.config.schema import ConfigurationError, validate_config

CONFIG_PATH = Path(__file__).parents[1] / "config" / "minimal.json"


@pytest.fixture
def valid_config():
    return json.loads(CONFIG_PATH.read_text())


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda cfg: cfg.pop("detector"), "config is missing required field 'detector'"),
        (
            lambda cfg: cfg["source"]["spectrum"].update(samples=0),
            "source.spectrum.samples must be a positive integer",
        ),
        (
            lambda cfg: cfg["source"]["spectrum"].update(Nu=4),
            "exactly one of 'samples' or 'Nu'",
        ),
        (
            lambda cfg: cfg["serial_elements"].update(Unknown_0={}),
            "unknown optical type 'Unknown'",
        ),
        (
            lambda cfg: cfg["serial_elements"]["CircularAperture_0"].update(
                nonsense=1
            ),
            "contains unknown field 'nonsense'",
        ),
    ],
)
def test_static_schema_errors_include_field_path(valid_config, mutation, message):
    config = deepcopy(valid_config)
    mutation(config)

    with pytest.raises(ConfigurationError, match=message):
        validate_config(config)


def test_invalid_band_is_reported_before_simulation(valid_config, tmp_path):
    valid_config["source"]["spectrum"]["band"] = "NOT_A_BAND"
    path = tmp_path / "invalid_band.json"
    path.write_text(json.dumps(valid_config))

    with pytest.raises(ConfigurationError, match="source.spectrum.band"):
        from_json(path)


def test_malformed_detector_is_reported_with_section_name(valid_config, tmp_path):
    valid_config["detector"]["exposure_time"] = -1
    path = tmp_path / "invalid_detector.json"
    path.write_text(json.dumps(valid_config))

    with pytest.raises(ConfigurationError, match="detector: exposure_time"):
        from_json(path)


def test_incompatible_detector_grid_is_rejected_before_run(valid_config, tmp_path):
    valid_config["detector"]["grid"] = "pupil_grid"
    path = tmp_path / "wrong_detector_grid.json"
    path.write_text(json.dumps(valid_config))

    with pytest.raises(ConfigurationError, match="final optical plane"):
        from_json(path)


def test_malformed_json_reports_location(tmp_path):
    path = tmp_path / "malformed.json"
    path.write_text('{"source": }')

    with pytest.raises(ConfigurationError, match="line 1, column"):
        from_json(path)
