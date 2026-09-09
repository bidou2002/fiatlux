import pytest
import torch

from fiatlux.core.grid import Grid
from fiatlux.optics.elements.mask import HarmoniResiduals


def make_residuals(monkeypatch, datacube, pupil=None):
    def load_datacube(instance, path):
        instance.datacube = datacube.clone()

    monkeypatch.setattr(HarmoniResiduals, "load_datacube", load_datacube)
    grid = Grid(nx=datacube.shape[-1], ny=datacube.shape[-2], dx=0.1, dy=0.1)
    return HarmoniResiduals(grid, pupil=pupil)


def test_public_pupil_uses_support_across_the_complete_cube(monkeypatch):
    datacube = torch.zeros((2, 3, 4))
    datacube[0, 0, 0] = 1.0
    datacube[1, 1, 2] = 1.0

    residuals = make_residuals(monkeypatch, datacube)

    assert residuals.pupil.dtype == torch.bool
    assert residuals.pupil[0, 0]
    assert residuals.pupil[1, 2]
    assert residuals.support is residuals.pupil


def test_explicit_pupil_is_independent_of_zero_opd_values(monkeypatch):
    datacube = torch.zeros((2, 3, 4))
    pupil = torch.zeros((3, 4), dtype=torch.bool)
    pupil[1, 1] = True

    residuals = make_residuals(monkeypatch, datacube, pupil=pupil)

    assert residuals.pupil[1, 1]
    assert torch.count_nonzero(residuals.pupil) == 1


def test_pupil_shape_and_binary_values_are_validated(monkeypatch):
    datacube = torch.zeros((2, 3, 4))

    with pytest.raises(ValueError, match="must have grid shape"):
        make_residuals(monkeypatch, datacube, pupil=torch.ones(4, 3))
    with pytest.raises(ValueError, match="only 0 and 1"):
        make_residuals(monkeypatch, datacube, pupil=torch.full((3, 4), 0.5))
