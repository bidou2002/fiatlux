from __future__ import annotations

import inspect
import math
from collections.abc import Mapping

from fiatlux.config.registry import TYPE_REGISTRY


class ConfigurationError(ValueError):
    """Invalid Fiatlux configuration with a path to the offending field."""


def _mapping(value, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a JSON object.")
    return value


def _fields(value, path: str, *, required, allowed):
    mapping = _mapping(value, path)
    missing = sorted(set(required) - mapping.keys())
    unknown = sorted(mapping.keys() - set(allowed))
    if missing:
        raise ConfigurationError(f"{path} is missing required field '{missing[0]}'.")
    if unknown:
        raise ConfigurationError(f"{path} contains unknown field '{unknown[0]}'.")
    return mapping


def _positive_integer(value, path: str):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigurationError(f"{path} must be a positive integer.")


def _finite_number(value, path: str):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ConfigurationError(f"{path} must be a finite number.")


def _validate_grid(config, path: str):
    config = _fields(
        config,
        path,
        required={"nx", "ny", "dx", "dy"},
        allowed={"nx", "ny", "dx", "dy", "device"},
    )
    _positive_integer(config["nx"], f"{path}.nx")
    _positive_integer(config["ny"], f"{path}.ny")
    for name in ("dx", "dy"):
        _finite_number(config[name], f"{path}.{name}")
        if config[name] <= 0:
            raise ConfigurationError(f"{path}.{name} must be positive.")
    if "device" in config and not isinstance(config["device"], str):
        raise ConfigurationError(f"{path}.device must be a string.")


def _validate_spectrum(config):
    config = _fields(
        config,
        "source.spectrum",
        required={"magnitude", "band"},
        allowed={"magnitude", "band", "samples", "Nu"},
    )
    _finite_number(config["magnitude"], "source.spectrum.magnitude")
    if not isinstance(config["band"], str):
        raise ConfigurationError("source.spectrum.band must be a string.")
    sampling_fields = [name for name in ("samples", "Nu") if name in config]
    if len(sampling_fields) != 1:
        raise ConfigurationError(
            "source.spectrum must contain exactly one of 'samples' or 'Nu'."
        )
    name = sampling_fields[0]
    _positive_integer(config[name], f"source.spectrum.{name}")


def _validate_element(name: str, params):
    path = f"serial_elements.{name}"
    if not isinstance(name, str):
        raise ConfigurationError("serial_elements names must be strings.")
    type_name = name.split("_")[0]
    if type_name not in TYPE_REGISTRY:
        raise ConfigurationError(f"{path} has unknown optical type '{type_name}'.")
    params = _mapping(params, path)
    signature = inspect.signature(TYPE_REGISTRY[type_name])
    constructor_parameters = {
        key: parameter
        for key, parameter in signature.parameters.items()
        if key != "self"
        and parameter.kind
        not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    }
    unknown = sorted(params.keys() - constructor_parameters.keys())
    if unknown:
        raise ConfigurationError(f"{path} contains unknown field '{unknown[0]}'.")
    required = {
        key
        for key, parameter in constructor_parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    missing = sorted(required - params.keys())
    if missing:
        raise ConfigurationError(f"{path} is missing required field '{missing[0]}'.")


def validate_config(config) -> None:
    """Validate the static structure and primitive types of a system config."""
    config = _fields(
        config,
        "config",
        required={"source", "pupil_grid", "focal_grid", "serial_elements", "detector"},
        allowed={"source", "pupil_grid", "focal_grid", "serial_elements", "detector"},
    )
    source = _fields(
        config["source"], "source", required={"spectrum"}, allowed={"spectrum"}
    )
    _validate_spectrum(source["spectrum"])
    _validate_grid(config["pupil_grid"], "pupil_grid")
    _validate_grid(config["focal_grid"], "focal_grid")

    elements = _mapping(config["serial_elements"], "serial_elements")
    if not elements:
        raise ConfigurationError("serial_elements must contain at least one element.")
    for name, params in elements.items():
        _validate_element(name, params)

    detector = _mapping(config["detector"], "detector")
    if "grid" not in detector:
        raise ConfigurationError("detector is missing required field 'grid'.")
    if not isinstance(detector["grid"], str):
        raise ConfigurationError("detector.grid must be an object-reference string.")
