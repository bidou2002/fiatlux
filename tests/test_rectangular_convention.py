import pytest
import torch

from fiatlux.core.field import Field
from fiatlux.core.grid import Grid
from fiatlux.core.source import PlaneWave
from fiatlux.core.spectrum import Band, Spectrum
from fiatlux.optics.detector import Detector
from fiatlux.optics.elements.field_stop import ShanonFieldStop
from fiatlux.optics.elements.mask import CircularAperture, TipTilt
from fiatlux.optics.propagator import MFTPropagator
from fiatlux.system.optical_system import SerialSystem


def make_spectrum(samples=2, wavelength=1e-6):
    return Spectrum(
        magnitude=0,
        band=Band(wavelength, 0.1e-6 if samples > 1 else 0.0, 1e8),
        samples=samples,
    )


def test_grid_coordinates_follow_y_x_array_order():
    grid = Grid(nx=7, ny=5, dx=0.2, dy=0.3)

    x, y = grid.meshgrid()

    assert grid.shape == (5, 7)
    assert x.shape == y.shape == grid.shape
    torch.testing.assert_close(x[0], grid.x)
    torch.testing.assert_close(y[:, 0], grid.y)
    assert torch.equal(x[0], x[-1])
    assert torch.equal(y[:, 0], y[:, -1])


def test_field_rejects_noncanonical_shapes():
    grid = Grid(nx=7, ny=5, dx=0.2, dy=0.3)
    spectrum = make_spectrum(samples=2)

    with pytest.raises(ValueError, match="n_wavelengths, ny, nx"):
        Field(torch.ones((2, 7, 5)), grid, spectrum)
    with pytest.raises(ValueError, match="n_wavelengths, ny, nx"):
        Field(torch.ones((5, 7)), grid, spectrum)


def test_tip_and_tilt_vary_along_the_named_axes():
    grid = Grid(nx=7, ny=5, dx=0.2, dy=0.3)
    spectrum = make_spectrum(samples=1)
    tip_only = TipTilt(grid, tip=1.0, tilt=0.0)
    tilt_only = TipTilt(grid, tip=0.0, tilt=1.0)

    tip_only.build(spectrum)
    tilt_only.build(spectrum)

    assert torch.all(torch.diff(tip_only.opd, dim=-1) > 0)
    assert torch.all(torch.diff(tip_only.opd, dim=-2) == 0)
    assert torch.all(torch.diff(tilt_only.opd, dim=-2) > 0)
    assert torch.all(torch.diff(tilt_only.opd, dim=-1) == 0)


def test_chromatic_field_stop_uses_lambda_y_x_order():
    grid = Grid(nx=7, ny=5, dx=0.2, dy=0.3)
    spectrum = make_spectrum(samples=2)
    field_stop = ShanonFieldStop(grid)

    field_stop.build(spectrum)

    assert field_stop.transmission.shape == (2, 5, 7)
    assert field_stop.opd.shape == (5, 7)


def test_rectangular_mft_has_output_y_x_shape():
    wavelength = 1e-6
    focal_length = 2.0
    input_grid = Grid(nx=8, ny=6, dx=0.1, dy=0.2)
    output_grid = Grid(nx=10, ny=4, dx=1e-6, dy=2e-6)
    field = PlaneWave(make_spectrum(samples=1, wavelength=wavelength)).generate_field(
        input_grid
    )

    propagated = MFTPropagator(focal_length, output_grid).apply(field)

    assert propagated.complex_amplitude.shape == (1, 4, 10)


@pytest.mark.parametrize(
    "tip_bin, tilt_bin",
    [(1, 0), (0, 1)],
)
def test_mft_tip_tilt_shift_psf_along_the_expected_axis(tip_bin, tilt_bin):
    nx, ny = 8, 6
    dx, dy = 0.1, 0.2
    wavelength = 1e-6
    focal_length = 2.0
    input_grid = Grid(nx=nx, ny=ny, dx=dx, dy=dy)
    output_grid = Grid(
        nx=nx,
        ny=ny,
        dx=wavelength * focal_length / (nx * dx),
        dy=wavelength * focal_length / (ny * dy),
    )
    spectrum = make_spectrum(samples=1, wavelength=wavelength)
    x, y = input_grid.meshgrid()
    phase = 2 * torch.pi * (
        tip_bin * x / (nx * dx) + tilt_bin * y / (ny * dy)
    )
    amplitude = torch.exp(1j * phase)[None]
    field = Field(amplitude, input_grid, spectrum)

    image = MFTPropagator(focal_length, output_grid).apply(field).intensity()[0]
    peak_y, peak_x = torch.unravel_index(image.argmax(), image.shape)

    assert peak_x.item() == nx // 2 + tip_bin
    assert peak_y.item() == ny // 2 + tilt_bin


def test_full_rectangular_simulation_reaches_detector():
    wavelength = 1e-6
    focal_length = 2.0
    input_grid = Grid(nx=8, ny=6, dx=0.1, dy=0.2)
    output_grid = Grid(nx=10, ny=4, dx=1e-6, dy=2e-6)
    source = PlaneWave(make_spectrum(samples=2, wavelength=wavelength))
    aperture = CircularAperture(input_grid, radius=0.4)
    propagator = MFTPropagator(focal_length, output_grid)
    detector = Detector(output_grid, exposure_time=1.0)
    system = SerialSystem([aperture, propagator])

    result = system.run(source, detector)

    assert result.field_at(aperture).complex_amplitude.shape == (2, 6, 8)
    assert result.field_at(propagator).complex_amplitude.shape == (2, 4, 10)
    assert detector.image_buffer.shape == (4, 10)
