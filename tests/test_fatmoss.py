import types
from itertools import islice

import numpy as np
import pytest
import torch

from fiatlux import (
    FatmossAtmosphereModel,
    FatmossUnavailableError,
    FrozenFlowLayer,
    Grid,
)


class FakeBackend:
    def __init__(self):
        self.requested = []

    def GetScreenByTimestep(self, timestep):
        self.requested.append(timestep)
        return np.arange(6, dtype=np.float32).reshape(3, 2) + 10 * timestep


class TranslatingBackend:
    def __init__(self, **kwargs):
        self.settings = kwargs
        self.layers = []
        self.requested = []
        self.reset_count = 0
        self.size = round(kwargs["D"] / kwargs["dx"])

    def AddLayer(self, layer):
        self.layers.append(layer)

    def GetScreenByTimestep(self, timestep):
        self.requested.append(timestep)
        layer = self.layers[0]
        pixels = layer.wind_speed * self.settings["dt"] * timestep / self.settings["dx"]
        x = np.arange(self.size, dtype=float)[:, None]
        return np.cos(2 * np.pi * (x - pixels) / self.size) * np.ones(
            (1, self.size)
        )

    def reset(self, regenerate_layers=True):
        assert regenerate_layers is True
        self.reset_count += 1


class WeightedBackend(TranslatingBackend):
    def GetScreenByTimestep(self, timestep):
        self.requested.append(timestep)
        integrated_nm = sum(layer.weight * layer.wind_speed for layer in self.layers)
        return np.full((self.size, self.size), integrated_nm + timestep)


def fake_layer_module():
    def layer(*args):
        names = (
            "weight",
            "altitude",
            "wind_speed",
            "wind_direction",
            "boiling_factor",
            "PSD_spatial_func",
            "PSD_temporal_func",
        )
        return types.SimpleNamespace(**dict(zip(names, args)))

    return types.SimpleNamespace(
        Layer=layer,
        vonKarmanPSD=lambda frequency, r0, outer_scale, wavelength: (
            frequency,
            r0,
            outer_scale,
            wavelength,
        ),
        SimpleBoiling=lambda frequency, sampling: (frequency, sampling),
    )


def test_adapter_converts_axes_nanometres_dtype_and_piston():
    grid = Grid(3, 2, 0.1, 0.2, dtype=torch.float64)
    backend = FakeBackend()
    model = FatmossAtmosphereModel(grid, backend)

    first = model.sample_opd(remove_piston=False)
    second = model.sample_opd(remove_piston=True)

    expected = torch.arange(6, dtype=torch.float64).reshape(3, 2).T * 1e-9
    torch.testing.assert_close(first, expected)
    assert second.mean().abs() < 1e-20
    assert backend.requested == [0, 1]
    assert first.shape == grid.shape
    assert first.dtype == grid.dtype


def test_seek_selects_an_absolute_backend_timestep():
    backend = FakeBackend()
    model = FatmossAtmosphereModel(Grid(3, 2, 0.1, 0.2), backend)
    model.seek(17)
    model.sample_opd()
    assert backend.requested == [17]


def test_factory_translates_grid_time_and_precision():
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return FakeBackend()

    module = types.SimpleNamespace(PhaseScreensGenerator=factory)
    grid = Grid(8, 8, 0.25, 0.25, dtype=torch.float64)
    model = FatmossAtmosphereModel.create(
        grid,
        time_step=0.002,
        batch_size=16,
        n_cascades=2,
        seed=42,
        module=module,
    )

    assert model.backend is not None
    assert captured == {
        "D": 2.0,
        "dx": 0.25,
        "dt": 0.002,
        "batch_size": 16,
        "n_cascades": 2,
        "seed": 42,
        "double_precision": True,
    }


def test_factory_rejects_grids_unsupported_by_upstream():
    module = types.SimpleNamespace(PhaseScreensGenerator=lambda **kwargs: FakeBackend())
    with pytest.raises(ValueError, match="square grid"):
        FatmossAtmosphereModel.create(
            Grid(8, 6, 0.25, 0.25), time_step=0.001, module=module
        )


def test_missing_backend_fails_only_when_factory_is_invoked(monkeypatch):
    def unavailable(name):
        raise ImportError(name)

    monkeypatch.setattr("fiatlux.optics.fatmoss.importlib.import_module", unavailable)
    with pytest.raises(FatmossUnavailableError, match="FATMOSS is optional"):
        FatmossAtmosphereModel.create(Grid(8, 8, 0.25, 0.25), time_step=0.001)


def test_frozen_flow_factory_maps_physical_layer_without_rounding_motion():
    grid = Grid(8, 8, 0.2, 0.2, dtype=torch.float64)
    backend_holder = {}

    def factory(**kwargs):
        backend_holder["backend"] = TranslatingBackend(**kwargs)
        return backend_holder["backend"]

    model = FatmossAtmosphereModel.create_frozen_flow(
        grid,
        [
            FrozenFlowLayer(
                r0=0.15,
                outer_scale=25.0,
                wind_speed=1.0,
                wind_direction=37.0,
                weight=0.8,
                altitude=1200.0,
            )
        ],
        time_step=0.1,
        seed=23,
        reference_wavelength=600e-9,
        phase_generator_module=types.SimpleNamespace(PhaseScreensGenerator=factory),
        layer_module=fake_layer_module(),
    )

    backend = backend_holder["backend"]
    layer = backend.layers[0]
    assert backend.settings["seed"] == 23
    assert (layer.weight, layer.altitude) == (0.8, 1200.0)
    assert (layer.wind_speed, layer.wind_direction) == (1.0, 37.0)
    assert layer.boiling_factor == 0.0
    assert layer.PSD_spatial_func("f") == ("f", 0.15, 25.0, 600.0)
    assert layer.PSD_temporal_func("f") == ("f", 0.2)

    # v * dt / dx = 0.5 pixel: the cadence is passed through continuously.
    x = np.arange(8, dtype=float)[:, None]
    expected_nm = np.cos(2 * np.pi * (x - 0.5) / 8) * np.ones((1, 8))
    expected = torch.as_tensor(expected_nm.T * 1e-9, dtype=torch.float64)
    torch.testing.assert_close(model.opd_at(0.1, remove_piston=False), expected)


def test_frozen_flow_time_is_independent_from_optical_sampling():
    grid = Grid(8, 8, 0.2, 0.2)
    backend = TranslatingBackend(
        D=1.6,
        dx=0.2,
        dt=0.05,
        batch_size=4,
        n_cascades=1,
        seed=4,
        double_precision=False,
    )
    backend.AddLayer(types.SimpleNamespace(wind_speed=0.5))
    model = FatmossAtmosphereModel(
        grid, backend, advance_after_sample=False, time_step=0.05
    )

    first = model.sample_opd(remove_piston=False)
    torch.testing.assert_close(model.sample_opd(remove_piston=False), first)
    assert model.current_time == 0.0

    model.advance()
    assert model.current_time == pytest.approx(0.05)
    assert not torch.equal(model.current_opd(remove_piston=False), first)
    with pytest.raises(ValueError, match="integer multiple"):
        model.seek_time(0.075)


def test_sequence_and_phase_queries_have_explicit_state_semantics():
    grid = Grid(8, 8, 0.2, 0.2)
    backend = TranslatingBackend(
        D=1.6,
        dx=0.2,
        dt=0.05,
        batch_size=4,
        n_cascades=1,
        seed=4,
        double_precision=False,
    )
    backend.AddLayer(types.SimpleNamespace(wind_speed=0.5))
    model = FatmossAtmosphereModel(
        grid, backend, advance_after_sample=False, time_step=0.05
    )

    screens = model.sequence_opd(3, advance=False, remove_piston=False)
    assert screens.shape == (3, *grid.shape)
    assert model.timestep == 0
    phase = model.phase_at(0.1, remove_piston=False)
    torch.testing.assert_close(
        phase,
        screens[2] * (2 * np.pi / model.reference_wavelength),
    )
    assert model.timestep == 0

    model.sequence_opd(3, advance=True)
    assert model.timestep == 3


def test_multilayer_relative_weights_are_normalized_with_fatmoss_semantics():
    grid = Grid(8, 8, 0.2, 0.2)
    backend_holder = {}

    def factory(**kwargs):
        backend_holder["backend"] = WeightedBackend(**kwargs)
        return backend_holder["backend"]

    model = FatmossAtmosphereModel.create_frozen_flow(
        grid,
        [
            FrozenFlowLayer(0.15, 25.0, 5.0, 0.0, weight=2.0, altitude=0.0),
            FrozenFlowLayer(0.15, 25.0, 12.0, 90.0, weight=8.0, altitude=9000.0),
        ],
        time_step=0.001,
        normalize_weights=True,
        phase_generator_module=types.SimpleNamespace(PhaseScreensGenerator=factory),
        layer_module=fake_layer_module(),
    )

    weights = [layer.weight for layer in model.backend.layers]
    assert weights == pytest.approx([0.2, 0.8])
    assert sum(weights) == pytest.approx(1.0)
    assert [layer.altitude for layer in model.backend.layers] == [0.0, 9000.0]
    assert [layer.wind_speed for layer in model.backend.layers] == [5.0, 12.0]
    assert [layer.wind_direction for layer in model.backend.layers] == [0.0, 90.0]
    expected_integrated_opd = (0.2 * 5.0 + 0.8 * 12.0) * 1e-9
    assert model.current_opd(remove_piston=False).mean().item() == pytest.approx(
        expected_integrated_opd
    )


def test_reset_is_deterministic_and_lazy_iteration_does_not_build_a_cube():
    grid = Grid(8, 8, 0.2, 0.2)
    backend = TranslatingBackend(
        D=1.6,
        dx=0.2,
        dt=0.05,
        batch_size=2,
        n_cascades=1,
        seed=4,
        double_precision=False,
    )
    backend.AddLayer(types.SimpleNamespace(wind_speed=0.5))
    model = FatmossAtmosphereModel(
        grid, backend, advance_after_sample=False, time_step=0.05
    )

    reference = model.current_opd(remove_piston=False)
    model.advance(37)
    assert not torch.equal(model.current_opd(remove_piston=False), reference)
    model.reset()
    torch.testing.assert_close(model.current_opd(remove_piston=False), reference)
    assert model.timestep == 0
    assert backend.reset_count == 1

    backend.requested.clear()
    iterator = model.iter_opd(number=1_000_000, remove_piston=False)
    first_three = list(islice(iterator, 3))
    assert len(first_three) == 3
    assert all(screen.dtype == grid.dtype for screen in first_three)
    assert all(screen.device == grid.device for screen in first_three)
    assert backend.requested == [0, 1, 2]
    assert model.timestep == 3
    next(iterator)
    assert model.timestep == 4
    assert backend.requested == [0, 1, 2, 3]
