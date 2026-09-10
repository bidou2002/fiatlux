import torch

from fiatlux.core.field import Field
from fiatlux.core.grid import Grid
from fiatlux.core.spectrum import Band, Spectrum
from fiatlux.optics.propagator import FFTPropagator, MFTPropagator

WAVELENGTH = 632.8e-9


def make_field(amplitude, grid):
    spectrum = Spectrum(
        magnitude=0,
        band=Band(WAVELENGTH, 0.0, 1.0),
        samples=1,
        dtype=grid.dtype,
        device=grid.device,
    )
    return Field(amplitude.to(torch.complex128).unsqueeze(0), grid, spectrum)


def test_fresnel_gaussian_beam_radius_curvature_and_gouy_phase():
    grid = Grid(256, 256, 15e-6, 15e-6, dtype=torch.float64)
    x, y = grid.meshgrid()
    radius_squared = x.square() + y.square()
    waist = 0.15e-3
    distance = 0.08
    field = make_field(torch.exp(-radius_squared / waist**2), grid)

    propagated = FFTPropagator(
        propagation="fresnel", distance=distance
    ).apply(field)
    intensity = propagated.intensity()[0]
    x_out, y_out = propagated.grid.meshgrid()
    output_radius_squared = x_out.square() + y_out.square()

    rayleigh_distance = torch.pi * waist**2 / WAVELENGTH
    expected_waist = waist * torch.sqrt(
        torch.as_tensor(
            1 + (distance / rayleigh_distance) ** 2, dtype=grid.dtype
        )
    )
    measured_waist = torch.sqrt(
        2 * (output_radius_squared * intensity).sum() / intensity.sum()
    )
    torch.testing.assert_close(measured_waist, expected_waist, rtol=3e-3, atol=0)

    expected_curvature = distance * (1 + (rayleigh_distance / distance) ** 2)
    center = propagated.grid.ny // 2
    line = propagated.complex_amplitude[0, center]
    relative_phase = torch.angle(line * line[center].conj())
    expected_phase = (
        torch.pi * propagated.grid.x.square() / (WAVELENGTH * expected_curvature)
    )
    core = propagated.grid.x.abs() <= expected_waist
    wrapped_error = torch.angle(
        torch.exp(1j * (relative_phase[core] - expected_phase[core]))
    )
    assert wrapped_error.abs().max() < 2e-2

    expected_gouy = torch.atan(
        torch.tensor(distance / rayleigh_distance, dtype=torch.float64)
    )
    expected_center_phase = 2 * torch.pi * distance / WAVELENGTH - expected_gouy
    center_phase_error = torch.angle(
        torch.exp(1j * (torch.angle(line[center]) - expected_center_phase))
    )
    assert center_phase_error.abs() < 2e-2


def test_circular_aperture_on_axis_fresnel_intensity_matches_analytic_result():
    grid = Grid(512, 512, 2.5e-6, 2.5e-6, dtype=torch.float64)
    x, y = grid.meshgrid()
    radius = 0.30e-3
    distance = 0.20
    aperture = (x.square() + y.square() <= radius**2).to(torch.float64)
    field = make_field(aperture, grid)
    origin = Grid(1, 1, 1e-6, 1e-6, dtype=torch.float64)

    propagated = MFTPropagator(
        output_grid=origin,
        propagation="fresnel",
        distance=distance,
    ).apply(field)

    fresnel_phase = torch.as_tensor(
        torch.pi * radius**2 / (WAVELENGTH * distance), dtype=grid.dtype
    )
    expected_intensity = 4 * torch.sin(fresnel_phase / 2).square()
    measured_intensity = propagated.intensity()[0, 0, 0]
    torch.testing.assert_close(
        measured_intensity, expected_intensity, rtol=2e-2, atol=2e-2
    )


def test_fresnel_intensity_converges_to_fraunhofer_in_far_field():
    grid = Grid(128, 128, 8e-6, 8e-6, dtype=torch.float64)
    x, y = grid.meshgrid()
    radius = 0.10e-3
    distance = 10.0
    aperture = (x.square() + y.square() <= radius**2).to(torch.float64)
    field = make_field(aperture, grid)

    fresnel = FFTPropagator(
        propagation="fresnel", distance=distance
    ).apply(field)
    fraunhofer = FFTPropagator(focal_length=distance).apply(field)
    fresnel_intensity = fresnel.intensity()[0]
    fraunhofer_intensity = fraunhofer.intensity()[0]
    fresnel_intensity = fresnel_intensity / fresnel_intensity.max()
    fraunhofer_intensity = fraunhofer_intensity / fraunhofer_intensity.max()

    relative_error = torch.linalg.vector_norm(
        fresnel_intensity - fraunhofer_intensity
    ) / torch.linalg.vector_norm(fraunhofer_intensity)
    assert relative_error < 5e-3


def test_forward_then_backward_fresnel_recovers_complex_field_and_grid():
    grid = Grid(64, 48, 20e-6, 30e-6, dtype=torch.float64)
    generator = torch.Generator().manual_seed(41)
    amplitude = torch.complex(
        torch.randn(grid.shape, generator=generator, dtype=torch.float64),
        torch.randn(grid.shape, generator=generator, dtype=torch.float64),
    )
    field = make_field(amplitude, grid)
    distance = 0.25

    forward = FFTPropagator(
        propagation="fresnel", distance=distance
    ).apply(field)
    recovered = FFTPropagator(
        propagation="fresnel", distance=-distance
    ).apply(forward)

    torch.testing.assert_close(
        recovered.complex_amplitude,
        field.complex_amplitude,
        rtol=2e-12,
        atol=2e-12,
    )
    assert recovered.grid.shape == field.grid.shape
    assert abs(recovered.grid.dx - field.grid.dx) < 1e-15
    assert abs(recovered.grid.dy - field.grid.dy) < 1e-15
