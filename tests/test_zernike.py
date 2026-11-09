import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
import torch

from fiatlux.utils.zernike import R, noll_indices, zernike, zernike_basis
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


def decimal_radial_reference(n, m, radii):
    """Independent defining polynomial with exact integer coefficients.

    100 decimal digits resolve cancellation at radial order 80. Deliberately
    do not use the production Jacobi recurrence for the reference.
    """
    with localcontext() as context:
        context.prec = 100
        terms = [
            (
                (-1)**k * math.factorial(n-k) // (
                    math.factorial(k)
                    * math.factorial((n+m)//2-k)
                    * math.factorial((n-m)//2-k)
                ),
                n-2*k,
            )
            for k in range((n-m)//2+1)
        ]
        return [float(sum(
            Decimal(coefficient) * (
                Decimal.from_float(radius)**power if power else Decimal(1)
            )
            for coefficient, power in terms
        )) for radius in radii]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_every_radial_mode_through_order_80_matches_independent_reference(dtype, device):
    radii = torch.tensor(
        [0., .001, .125, .37, .5, .75, .9, .99, 1.], dtype=dtype, device=device
    )
    # Reference the values actually represented by each input dtype.
    represented_radii = radii.tolist()
    tolerance = 1e-7 if dtype == torch.float32 else 5e-13
    for n in range(81):
        for m in range(n % 2, n+1, 2):
            expected = torch.tensor(
                decimal_radial_reference(n, m, represented_radii),
                dtype=dtype, device=device,
            )
            actual = R(n, m, radii)
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            assert actual.dtype == dtype and actual.device == radii.device
            torch.testing.assert_close(R(n, -m, radii), actual, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_high_order_noll_normalization_by_independent_disk_quadrature(dtype):
    # Degree <= 80: 96-point Gauss--Legendre integrates R**2 * rho
    # exactly in exact arithmetic (degree <= 161). Angular means are 1 or
    # 1/2, so this checks continuous Noll normalization without pixel bias.
    nodes, weights = np.polynomial.legendre.leggauss(96)
    radii = torch.tensor((nodes+1)/2, dtype=dtype)
    radial_weights = torch.tensor(weights/2, dtype=torch.float64)
    for n in range(81):
        for m in range(n % 2, n+1, 2):
            radial = R(n, m, radii).double()
            mean_square = 2*(n+1)*(radial_weights*radii.double()*radial.square()).sum()
            # Casting the quadrature nodes to float32 perturbs high-order
            # modes near rho=1; allow its O(n**2 * eps) quadrature effect.
            tolerance = 2e-5 if dtype == torch.float32 else 2e-12
            assert abs(float(mean_square)-1) < tolerance, (n, m, mean_square)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_high_order_angles_masking_and_coordinate_gradients(dtype, device):
    rho = torch.tensor([0., .1, .5, .9, 1., 1e6], dtype=dtype, device=device,
                       requires_grad=True)
    theta = torch.tensor([0., .2, -.7, 2.3, -3., .4], dtype=dtype, device=device,
                         requires_grad=True)
    for m in (20, -20):
        actual = zernike(80, m, rho=rho, theta=theta, outside=-7.)
        radial = torch.tensor(decimal_radial_reference(80, abs(m), rho[:-1].tolist()),
                              dtype=torch.float64, device=device)
        angular = torch.cos(abs(m)*theta[:-1].double()) if m > 0 else torch.sin(abs(m)*theta[:-1].double())
        expected = (math.sqrt(162)*radial*angular).to(dtype)
        tolerance = 1e-6 if dtype == torch.float32 else 5e-12
        torch.testing.assert_close(actual[:-1], expected, rtol=tolerance, atol=tolerance)
        assert actual[-1] == -7.
        gradients = torch.autograd.grad(actual.sum(), (rho, theta), retain_graph=True)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert all(gradient[-1] == 0 for gradient in gradients)
    assert torch.autograd.gradcheck(
        lambda r, t: zernike(80, -20, rho=r, theta=t),
        (rho.detach()[1:4].double().requires_grad_(), theta.detach()[1:4].double().requires_grad_()),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_3240_control_modes_on_128_grid_have_finite_noll_scale(dtype):
    grid = Grid(128, 128, 37/128, 37/128, dtype=dtype)
    basis = ZernikeBasis(grid, 3240)
    matrix = basis.build_command_matrix()
    rho, _ = basis._polar_coordinates()
    assert noll_indices(3241) == (80, 0)
    assert matrix.shape == (128*128, 3240)
    assert matrix.dtype == dtype and matrix.device == grid.device
    assert torch.isfinite(matrix).all()
    assert torch.count_nonzero(matrix[rho.flatten() > 1]) == 0
    # |R_n^m| <= 1 on the disk and |Z| <= sqrt(2*(n+1)).
    assert float(matrix.abs().max()) <= math.sqrt(162)*(1+1e-6)
    rms = matrix[rho.flatten() <= 1].square().mean(0).sqrt()
    assert torch.isfinite(rms).all() and torch.all(rms > 0)
    # Compare the sampled radial-80 mode to its 100-digit definition on the
    # *same* Cartesian grid, instead of forcing its discrete RMS to unity.
    radii, counts = torch.unique(rho[rho <= 1], return_counts=True)
    reference = torch.tensor(decimal_radial_reference(80, 0, radii.tolist()), dtype=torch.float64)
    reference_rms = math.sqrt(81)*(
        (reference.square()*counts.double()).sum()/counts.sum()
    ).sqrt()
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(rms[-1].double(), reference_rms, rtol=tolerance, atol=tolerance)
