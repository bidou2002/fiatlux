import types

import numpy as np
import pytest
import torch

from fiatlux import FatmossAtmosphereModel, FatmossUnavailableError, Grid


class FakeBackend:
    def __init__(self):
        self.requested = []

    def GetScreenByTimestep(self, timestep):
        self.requested.append(timestep)
        return np.arange(6, dtype=np.float32).reshape(3, 2) + 10 * timestep


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
