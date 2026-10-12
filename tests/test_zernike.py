import math

import pytest
import torch

from fiatlux.utils.zernike import R, zernike, zernike_basis
from fiatlux.core.grid import Grid
from fiatlux.optics.elements.deformable_mirror import ZernikeBasis

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_analytical_modes_and_precision(device, dtype):
    x = torch.tensor([0., .123456789123, -.3], device=device, dtype=dtype)
    y = torch.tensor([0., -.234567891234, .4], device=device, dtype=dtype)
    rho, theta = torch.hypot(x, y), torch.atan2(y, x)
    modes = zernike_basis(4, rho=rho, theta=theta)
    expected = torch.stack([torch.ones_like(x), 2*x, 2*y,
                            math.sqrt(3)*(2*rho.square()-1)])
    tolerance = 3e-14 if dtype == torch.float64 else 2e-6
    torch.testing.assert_close(modes, expected, rtol=tolerance, atol=tolerance)
    assert modes.dtype == dtype and modes.device == x.device
    assert torch.isfinite(modes).all()
    torch.testing.assert_close(zernike(1, 1, rho=rho, theta=theta,
                                     noll_normalize=False), x,
                               rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(R(1, 0, rho), torch.zeros_like(rho))


@pytest.mark.parametrize("n,m", [(0, 0), (1, 1), (1, -1), (2, 0), (3, 1)])
def test_noll_mean_square_on_refined_unit_disk(n, m):
    axis = torch.linspace(-1, 1, 501, dtype=torch.float64)
    x, y = torch.meshgrid(axis, axis, indexing="xy")
    rho, theta = torch.hypot(x, y), torch.atan2(y, x)
    mode = zernike(n, m, rho=rho, theta=theta)
    assert abs(mode[rho <= 1].square().mean().item()-1) < .003


def test_control_basis_constructs_in_float64():
    grid = Grid(19, 19, .1, .1, dtype=torch.float64)
    matrix = ZernikeBasis(grid, 3).build_command_matrix()
    assert matrix.dtype == torch.float64
    assert torch.isfinite(matrix).all()
