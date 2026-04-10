from fiatlux.optics.elements.mask import CircularAperture
from fiatlux.core.grid import Grid
from fiatlux.core.spectrum import *
from fiatlux.core.source import *
from fiatlux.optics.propagator import MFTPropagator, IdentityPropagator
from fiatlux.optics.elements.mask import (
    ZeldaMask,
    ZeldaStop,
    ADC,
    TipTilt,
    Piston,
    ArbitraryAperture,
)
from fiatlux.utils.converter import FocalPlaneConverter
from fiatlux.utils.fits_loader import load_pupil, load_zelda_measurement
from fiatlux.optics.elements.field_stop import ShanonFieldStop
from fiatlux.system.optical_system import SerialSystem
from fiatlux.optics.detector import Detector
from fiatlux.optics.elements.deformable_mirror import (
    DeformableMirror,
    ActuatorGrid,
    InteractionMatrix,
    GaussianZonalBasis,
    FourierBasis,
    SquareZonalBasis,
    SquarePTTZonalBasis,
    ZernikeBasis,
)
from fiatlux.optics.adc import ADCDispersionModel

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import json
from config.serial_config import SystemConfig


def forward_model(system: SerialSystem, source, commands, response_function):
    dm: DeformableMirror = system.elements[4]
    dm._commands = commands
    return response_function(system.run(source=source))


if __name__ == "__main__":

    # %% DEFINITIONS
    # Define source spectrum
    spectrum = Spectrum.from_sampling(magnitude=5, band=PhotometricBand.HCM, Nu=512)

    # System parameters
    f = 0.125
    lambda_max = spectrum.wavelengths.max()
    D = 38.542

    pupil_mask = load_pupil()

    # Pupil plane sampling
    Nx, Ny = pupil_mask.shape
    dx = dy = D / Nx
    pupil_grid = Grid(nx=Nx, ny=Ny, dx=dx, dy=dx)

    # Focal plane sampling
    Nu = Nv = 128
    # # Shanon
    # du = dv = Nx * (f * l / D) / Nu
    du = dv = (10) * (f * lambda_max / D) / (Nu / 2)
    focal_grid = Grid(nx=Nu, ny=Nv, dx=du, dy=dv)

    # Create source
    source = PlaneWave(spectrum=spectrum)

    # Empty propagator (for testing)
    empty_propagator0 = IdentityPropagator(pupil_grid)
    empty_propagator1 = IdentityPropagator(pupil_grid)

    # Create aperture
    aperture = ArbitraryAperture(torch.as_tensor(pupil_mask))

    adc_model = ADCDispersionModel()
    amplitudes = adc_model.get_dispersion(angle=30, spectrum=spectrum)
    print(amplitudes / (lambda_max / D))
    adc = ADC(amplitude=amplitudes, angle=90)

    tip_tilt_compensator = TipTilt(tip=0, tilt=-amplitudes.max() / 2)

    n_modes = 2
    basis = ZernikeBasis(pixel_grid=pupil_grid, n=n_modes)

    # DM
    dm = DeformableMirror(
        actuator_grid=ActuatorGrid(n_actuators_x=10, n_actuators_y=10, pitch=D / 10),
        pixel_grid=pupil_grid,
        control_basis=basis,
        stroke=lambda_max,
    )

    # Define propagator
    mft0 = MFTPropagator(focal_length=f, output_grid=focal_grid)
    mft1 = MFTPropagator(focal_length=f, output_grid=pupil_grid)

    theta = torch.as_tensor(lambda_max / 4)
    # Define zelda mask
    zelda_mask = ZeldaMask(radius=f * lambda_max / D, well_depth=theta)
    zelda_stop = ZeldaStop(radius=f * lambda_max / D)

    detector = Detector(
        grid=pupil_grid,
        photon_noise=False,
        readout_noise_variance=0,
        dark_current=0,
        offset=0,
    )

    system = SerialSystem(
        elements=[
            empty_propagator0,
            aperture,
            adc,
            tip_tilt_compensator,
            dm,
            mft0,
            zelda_stop,
            mft1,
            empty_propagator1,
        ],
    )

    def response_function(result):
        detector.acquire(
            (result.field_at(4) - result.field_at(-1))
            + torch.exp(1j * torch.as_tensor(torch.pi) / 2) * result.field_at(-1)
        )
        return detector.image_buffer

    # %% FIT
    torch.manual_seed(0)

    # INIT DM COMMANDS
    commands_0 = 2 * (torch.rand(n_modes) - 0.5) * 0.0
    dm._commands = commands_0

    res = system.run(source=source)
    print(*[(i, el) for i, el in enumerate(system.elements)], sep="\n")

    res.field_at(5).plot()
    plt.draw()

    res.field_at(6).plot()
    plt.draw()

    plt.figure()
    plt.imshow(response_function(res))
    plt.colorbar()
    plt.draw()

    reference_image = response_function(res)

    observed_image = load_zelda_measurement()
    observed_image = torch.tensor(
        observed_image.astype(observed_image.dtype.newbyteorder("="))
    )
    observed_image /= observed_image.sum()

    coeffs = torch.nn.Parameter(0.1 * torch.rand(dm.commands.shape, requires_grad=True))
    optimizer = torch.optim.Adam([coeffs], lr=1e-2)

    loss_plot = []

    n_iter = 100

    loss_fig, loss_ax = plt.subplots()

    for i in range(n_iter):

        optimizer.zero_grad()

        simulated_image = (
            forward_model(
                system, source, commands=coeffs, response_function=response_function
            )
            - reference_image
        )
        simulated_image /= simulated_image.sum()

        loss = torch.sum((simulated_image - observed_image) ** 2)

        loss.backward()

        optimizer.step()
        loss_plot += [loss.item()]

        print(i, f"{loss.item():.5e}", end="\r")

    loss_ax.plot(loss_plot)
    loss_ax.set_xlim(0, n_iter)
    loss_ax.set_yscale("log")
    loss_fig.suptitle("Loss")
    plt.draw()

    fig, ax = plt.subplots(1, 3)
    fig.suptitle("$I_{fit} - I_{true}$")
    ax[0].imshow((observed_image).detach())
    ax[0].set_title("Bench image")
    ax[1].imshow((simulated_image).detach())
    ax[1].set_title("Simulated image")
    ax[2].imshow(
        (simulated_image.detach() - observed_image.detach()) / observed_image.detach()
    )
    ax[2].set_title("Normalized difference")
    plt.draw()

    plt.show()
