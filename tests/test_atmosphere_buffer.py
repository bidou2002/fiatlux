"""Frozen-atmosphere end-to-end references; no independent turbulence draws."""

import importlib.util
import json
import types

import numpy as np
import pytest
import torch

from fiatlux import (
    ActuatorGrid, Atmosphere, DeformableMirror, Detector, ExposureAccumulator,
    FatmossAtmosphereModel, FFTPropagator, FrozenFlowLayer, Grid,
    KolmogorovAtmosphereModel, MFTPropagator, PlaneWave, SerialSystem, Spectrum,
    ZernikeBasis,
)
from fiatlux.core.spectrum import Band
from fiatlux.optics.turbulence_validation import temporal_autocorrelation
from test_fatmoss import TranslatingBackend, fake_layer_module
from test_latent_dimensions import assert_roundoff_close


def error_metrics(actual, expected):
    # Measure unscaled errors, using float64 for the diagnostic reduction only.
    delta = (actual - expected).abs().to(torch.float64)
    denominator = expected.abs().to(torch.float64).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    return {
        "max_absolute": float(delta.max()),
        "max_relative": float((delta / denominator).max()),
        "rms": float(delta.square().mean().sqrt()),
    }


def capture_cube(monkeypatch, model, method):
    """Record the exact samples consumed by apply_buffer without replacing sampling."""
    original = getattr(model, method)
    cubes = []

    def sample(*args, **kwargs):
        cube = original(*args, **kwargs)
        cubes.append(cube.detach().clone())
        return cube

    monkeypatch.setattr(model, method, sample)
    return cubes


def compare_pipeline(base, buffer, cube, system, dt):
    with torch.no_grad():
        # The reference only consumes the captured cube; never resample here.
        loop = torch.stack([
            system.run_field(base.apply_opd(screen)).final_field.complex_amplitude
            for screen in cube
        ])
        batched = system.run_field(buffer).final_field
        assert batched.dimensions == buffer.dimensions
        assert_roundoff_close(batched.complex_amplitude, loop)
        expected = (
            (loop.abs().square().sum(dim=-3) * dt).sum(dim=0)
            * batched.grid.dx * batched.grid.dy
        )
        detector = Detector(
            batched.grid, exposure_time=len(cube) * dt, quantum_efficiency=1,
            photon_noise=False, dark_current=0, readout_noise_variance=0,
        )
        counts = detector.acquire(batched, integrate_over="time")
        assert counts.shape == batched.grid.shape
        assert_roundoff_close(counts, expected)
        accumulator = ExposureAccumulator(detector)
        accumulator.add(batched.slice("time", 0, 1))
        accumulator.add(batched.slice("time", 1, len(cube)))
        chunked = accumulator.finish()
        assert_roundoff_close(chunked, expected)
    return {
        "field": error_metrics(batched.complex_amplitude, loop),
        "counts": error_metrics(counts, expected),
        "chunked": error_metrics(chunked, expected),
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", ["fft", "polychromatic_mft", "fresnel", "static_dm"])
def test_atmosphere_buffer_matches_frozen_loop(monkeypatch, record_property, dtype, case):
    # ZernikeBasis currently constructs square modes; other cases stay rectangular.
    grid = (
        Grid(8, 8, 0.1, 0.1, dtype=dtype)
        if case == "static_dm" else Grid(8, 6, 0.1, 0.12, dtype=dtype)
    )
    wavelengths = 3 if case == "polychromatic_mft" else 1
    spectrum = Spectrum(0, Band(1e-6, 0.2e-6, 368), wavelengths, dtype=dtype)
    base = PlaneWave(spectrum).generate_field(grid)
    model = KolmogorovAtmosphereModel(grid, 0.2, seed=902)
    cubes = capture_cube(monkeypatch, model, "sample_opd_many")
    atmosphere = Atmosphere(grid, model)
    state = model.generator.get_state().clone()
    dt, number = 0.007, 4  # Exposure != 1: an extra time multiplier cannot hide.
    buffer = atmosphere.apply_buffer(base, number, sample_period=dt, advance=False)
    assert torch.equal(model.generator.get_state(), state)
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.shape == (number, grid.ny, grid.nx)
    assert buffer.complex_amplitude.shape == (number, wavelengths, grid.ny, grid.nx)
    torch.testing.assert_close(
        buffer.complex_amplitude,
        base.apply_opd(cube, dimensions=buffer.dimensions).complex_amplitude,
        rtol=0, atol=0,
    )
    time = buffer.dimension("time")
    assert time.unit == "s"
    torch.testing.assert_close(time.coordinates, torch.arange(number, dtype=dtype) * dt)
    torch.testing.assert_close(
        time.integration_weights, torch.full((number,), dt, dtype=dtype)
    )
    output = Grid(7, 5, 1e-6, 1.2e-6, dtype=dtype)
    if case == "polychromatic_mft":
        elements = [MFTPropagator(2.0, output)]
    elif case == "fresnel":
        elements = [MFTPropagator(
            output_grid=output, propagation="fresnel", distance=2.0
        )]
    else:
        elements = [FFTPropagator(2.0)]
        if case == "static_dm":
            dm = DeformableMirror(
                grid, ActuatorGrid(3, 3, 0.1), grid, ZernikeBasis(grid, 3)
            )
            dm.commands = torch.tensor([20e-9, -10e-9, 5e-9], dtype=dtype)
            elements.insert(0, dm)
    metrics = compare_pipeline(base, buffer, cube, SerialSystem(elements), dt)
    record_property("errors", json.dumps(metrics))
    print(json.dumps({"case": case, "dtype": str(dtype), **metrics}))
    assert len(cubes) == 1  # Neither propagation path sampled the atmosphere again.


class SeededTranslatingBackend(TranslatingBackend):
    """Existing fractional-pixel backend with a reproducible nontrivial OPD scale."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.amplitude_nm = np.random.default_rng(kwargs["seed"]).uniform(40, 80)

    def GetScreenByTimestep(self, timestep):
        return self.amplitude_nm * super().GetScreenByTimestep(timestep)


@pytest.mark.parametrize("backend", ["injected", "real"])
def test_correlated_buffer_preserves_samples_time_and_replay(
    monkeypatch, record_property, backend
):
    kwargs = {}
    size = 8
    if backend == "real":
        if importlib.util.find_spec("phase_generator") is None:
            pytest.skip("Optional real FATMOSS source is not importable")
        size = 135  # Compatible with the supported upstream three-cascade grid.
    else:
        kwargs = {
            "phase_generator_module": types.SimpleNamespace(
                PhaseScreensGenerator=SeededTranslatingBackend
            ),
            "layer_module": fake_layer_module(),
        }
    grid = Grid(size, size, 0.1, 0.1, dtype=torch.float64)
    dt, number, start = 0.01, 4, 2
    model = FatmossAtmosphereModel.create_frozen_flow(
        grid, [FrozenFlowLayer(0.2, 25.0, 3.7, 0.0)],
        time_step=dt, batch_size=8, seed=51, **kwargs,
    )
    base = PlaneWave(
        Spectrum(0, Band(1e-6, 0, 368), 1, dtype=grid.dtype)
    ).generate_field(grid)
    atmosphere = Atmosphere(grid, model)
    cubes = capture_cube(monkeypatch, model, "sequence_opd")
    model.seek(start)
    buffer = atmosphere.apply_buffer(base, number, advance=False)
    assert model.timestep == start and model.current_time == start * dt
    cube = cubes[0]
    # Scalar backend sampling is a replay check; the optical reference uses cube.
    scalar_samples = torch.stack([
        model.opd_at((start + t) * dt) for t in range(number)
    ])
    torch.testing.assert_close(cube, scalar_samples, rtol=0, atol=0)
    torch.testing.assert_close(
        temporal_autocorrelation(cube), temporal_autocorrelation(scalar_samples),
        rtol=0, atol=0,
    )
    assert not torch.equal(cube[0], cube[1])
    torch.testing.assert_close(
        buffer.dimension("time").coordinates,
        start * dt + torch.arange(number, dtype=grid.dtype) * dt,
    )
    metrics = compare_pipeline(
        base, buffer, cube, SerialSystem([FFTPropagator(2.0)]), dt
    )
    assert model.timestep == start
    record_property("errors", json.dumps(metrics))
    print(json.dumps({"backend": backend, **metrics}))
    # Reset must replay the same seed, including from a nonzero selected instant.
    model.reset()
    model.seek(start)
    advanced = atmosphere.apply_buffer(base, number, advance=True)
    assert model.timestep == start + number
    assert model.current_time == (start + number) * dt
    torch.testing.assert_close(cubes[-1], cube, rtol=0, atol=0)
    torch.testing.assert_close(
        advanced.complex_amplitude, buffer.complex_amplitude, rtol=0, atol=0
    )
    # Chunking preserves the exact sequence, hence also its temporal correlation.
    model.reset()
    model.seek(start)
    atmosphere.apply_buffer(base, 1)
    atmosphere.apply_buffer(base, number - 1)
    torch.testing.assert_close(torch.cat(cubes[-2:]), cube, rtol=0, atol=0)
    assert model.timestep == start + number
