import matplotlib.pyplot as plt
import torch

from fiatlux.core.grid import Grid
from fiatlux.optics.atmosphere import (
    KolmogorovAtmosphereModel,
    NCPAModel,
)
from fiatlux.optics.elements.deformable_mirror import (
    ActuatorGrid,
    DeformableMirror,
    FourierBasis,
)
from fiatlux.optics.elements.mask import CircularAperture
from fiatlux.optics.propagator import MFTPropagator


def make_fourier_basis():
    grid = Grid(nx=12, ny=10, dx=0.1, dy=0.1)
    pupil = CircularAperture(grid, radius=0.45)
    pupil._build_transmission()
    return FourierBasis(
        pixel_grid=grid,
        frequencies=torch.tensor([-1.0, 0.0, 1.0]),
        pupil=pupil,
    )


def test_numerical_construction_and_building_open_no_figures():
    plt.close("all")
    initial_figures = set(plt.get_fignums())
    grid = Grid(nx=12, ny=10, dx=0.1, dy=0.1)

    KolmogorovAtmosphereModel(grid, r0=0.2, seed=1)
    NCPAModel(grid, opd_rms=100e-9, seed=1)
    basis = make_fourier_basis()
    basis.build_command_matrix()
    DeformableMirror(
        grid=basis.pixel_grid,
        actuator_grid=ActuatorGrid(1, 1, 1.0),
        pixel_grid=basis.pixel_grid,
        control_basis=basis,
    )
    MFTPropagator(focal_length=10.0, output_grid=grid)

    assert set(plt.get_fignums()) == initial_figures


def test_fourier_diagnostic_is_available_explicitly():
    basis = make_fourier_basis()

    figure, axes = basis.plot_mode_selection()

    assert figure.number in plt.get_fignums()
    assert axes.get_title() == "Fourier basis mode selection"
    plt.close(figure)


def test_atmosphere_and_ncpa_psd_diagnostics_are_explicit():
    grid = Grid(nx=12, ny=10, dx=0.1, dy=0.1)
    atmosphere = KolmogorovAtmosphereModel(grid, r0=0.2, seed=1)
    ncpa = NCPAModel(grid, opd_rms=100e-9, seed=1)

    atmosphere_figure, _ = atmosphere.plot_psd()
    ncpa_figure, _ = ncpa.plot_psd()

    assert atmosphere_figure.number in plt.get_fignums()
    assert ncpa_figure.number in plt.get_fignums()
    plt.close(atmosphere_figure)
    plt.close(ncpa_figure)
