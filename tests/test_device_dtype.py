import pytest
import torch

from fiatlux.core.field import Field
from fiatlux.core.grid import Grid
from fiatlux.core.source import PlaneWave
from fiatlux.core.spectrum import Band, Spectrum
from fiatlux.optics.detector import Detector
from fiatlux.optics.elements.deformable_mirror import (
    ActuatorGrid,
    DeformableMirror,
    SquareZonalBasis,
)
from fiatlux.optics.elements.mask import CircularAperture, TipTilt
from fiatlux.optics.propagator import MFTPropagator


def make_spectrum(device="cpu", dtype=torch.float32, samples=2):
    return Spectrum(
        magnitude=0,
        band=Band(1.0e-6, 0.1e-6, 1.0e8),
        samples=samples,
        device=device,
        dtype=dtype,
    )


def run_system(device, real_dtype, samples=2):
    pupil_grid = Grid(8, 6, 0.1, 0.12, device=device, dtype=real_dtype)
    image_grid = Grid(10, 4, 1.0e-6, 1.5e-6, device=device, dtype=real_dtype)
    spectrum = make_spectrum(device, real_dtype, samples=samples)
    field = PlaneWave(spectrum).generate_field(pupil_grid)
    field = CircularAperture(pupil_grid, radius=0.31).apply(field)
    field = TipTilt(pupil_grid, tip=1.0e-8, tilt=-2.0e-8).apply(field)
    field = MFTPropagator(2.0, image_grid).apply(field)
    image = Detector(image_grid, exposure_time=1.0).acquire(field)
    return field, image


@pytest.mark.parametrize(
    "real_dtype,complex_dtype",
    [(torch.float32, torch.complex64), (torch.float64, torch.complex128)],
)
def test_representative_system_preserves_device_and_precision(
    real_dtype, complex_dtype
):
    field, image = run_system("cpu", real_dtype)

    assert field.grid.dtype == real_dtype
    assert field.spectrum.wavelengths.dtype == real_dtype
    assert field.spectrum.fluxes.dtype == real_dtype
    assert field.complex_amplitude.dtype == complex_dtype
    assert image.dtype == real_dtype
    assert field.complex_amplitude.device.type == "cpu"
    assert image.device.type == "cpu"


def test_float32_and_float64_systems_agree_numerically():
    _, image32 = run_system("cpu", torch.float32, samples=1)
    _, image64 = run_system("cpu", torch.float64, samples=1)

    torch.testing.assert_close(image32.double(), image64, rtol=2e-5, atol=1e-8)


def test_field_to_transfers_all_associated_state():
    grid = Grid(5, 3, 0.1, 0.2)
    field = PlaneWave(make_spectrum()).generate_field(grid)

    moved = field.to(dtype=torch.complex128)

    assert moved.complex_amplitude.dtype == torch.complex128
    assert moved.grid.dtype == torch.float64
    assert moved.spectrum.wavelengths.dtype == torch.float64
    assert moved.spectrum.fluxes.dtype == torch.float64


def test_deformable_mirror_uses_grid_device_and_dtype():
    grid = Grid(7, 5, 0.1, 0.1, dtype=torch.float64)
    actuators = ActuatorGrid(2, 2, 0.2)
    basis = SquareZonalBasis(actuators, grid, influence_width=0.11)
    dm = DeformableMirror(grid, actuators, grid, basis)

    assert dm.commands.dtype == torch.float64
    assert dm.commands.device == grid.device
    assert dm._command_matrix.dtype == torch.float64
    assert dm._command_matrix.device == grid.device
    assert dm.opd.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_and_cuda_systems_agree():
    _, cpu_image = run_system("cpu", torch.float64)
    cuda_field, cuda_image = run_system("cuda", torch.float64)

    assert cuda_field.complex_amplitude.device.type == "cuda"
    assert cuda_image.device.type == "cuda"
    torch.testing.assert_close(cuda_image.cpu(), cpu_image, rtol=1e-10, atol=1e-10)


def test_field_rejects_unsupported_transfer_dtype():
    field = Field(
        torch.ones((2, 3, 5), dtype=torch.complex64),
        Grid(5, 3, 0.1, 0.2),
        make_spectrum(),
    )

    with pytest.raises(ValueError, match="Field dtype"):
        field.to(dtype=torch.float16)
