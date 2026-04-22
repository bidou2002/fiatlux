from fiatlux.optics.elements.mask import CircularAperture
from fiatlux.core.grid import Grid
from fiatlux.core.spectrum import *
from fiatlux.core.source import *
from fiatlux.optics.propagator import MFTPropagator, IdentityPropagator
from fiatlux.optics.elements.mask import ZeldaMask, ZeldaStop, ADC, TipTilt, Piston
from fiatlux.utils.converter import FocalPlaneConverter
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
    # Define source spectrum
    spectrum = Spectrum.from_sampling(magnitude=12, band=PhotometricBand.HCM2, Nu=512)

    # System parameters
    f = 0.125
    lambda_max = spectrum.wavelengths.max()
    D = 38.542

    # Pupil plane sampling
    Nx = Ny = 128
    dx = dy = D / Nx
    pupil_grid = Grid(nx=Nx, ny=Ny, dx=dx, dy=dx)

    # Focal plane sampling
    Nu = Nv = 512
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
    aperture = CircularAperture(grid=pupil_grid, radius=D / 2)

    adc_model = ADCDispersionModel()
    amplitudes = adc_model.get_dispersion(angle=53, spectrum=spectrum)
    print(amplitudes / (lambda_max / D))
    adc = ADC(grid=pupil_grid, amplitude=amplitudes, angle=90)

    tip_tilt_compensator = TipTilt(grid=pupil_grid, tip=0, tilt=-amplitudes.max() / 2)

    n_modes = 3
    # actuator_grid = ActuatorGrid(
    #     n_actuators_x=n_modes, n_actuators_y=n_modes, pitch=D / n_modes
    # )
    # basis = GaussianZonalBasis(
    #     actuator_grid=actuator_grid,
    #     pixel_grid=pupil_grid,
    #     influence_width=1,
    # )
    basis = ZernikeBasis(pixel_grid=pupil_grid, n=n_modes)

    # DM
    dm = DeformableMirror(
        grid=pupil_grid,
        actuator_grid=ActuatorGrid(n_actuators_x=10, n_actuators_y=10, pitch=D / 10),
        pixel_grid=pupil_grid,
        control_basis=basis,
        stroke=lambda_max,
    )
    ncpa_dm = DeformableMirror(
        grid=pupil_grid,
        actuator_grid=ActuatorGrid(n_actuators_x=10, n_actuators_y=10, pitch=D / 10),
        pixel_grid=pupil_grid,
        control_basis=ZernikeBasis(pixel_grid=pupil_grid, n=100),
        stroke=lambda_max,
    )

    # Define propagator
    mft0 = MFTPropagator(focal_length=f, output_grid=focal_grid)
    mft1 = MFTPropagator(focal_length=f, output_grid=pupil_grid)

    theta = torch.as_tensor(lambda_max / 4)
    # Define zelda mask
    zelda_mask = ZeldaMask(grid=focal_grid, radius=f * lambda_max / D, well_depth=theta)
    zelda_stop = ZeldaStop(grid=focal_grid, radius=f * lambda_max / D)

    torch.manual_seed(0)

    detector = Detector(
        grid=pupil_grid,
        photon_noise=True,
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
            ncpa_dm,
            mft0,
            zelda_stop,
            mft1,
            empty_propagator1,
        ],
    )

    def response_function(result):
        detector.acquire(
            (result.field_at(5) - result.field_at(-1))
            + torch.exp(1j * torch.as_tensor(torch.pi) / 2) * result.field_at(-1)
        )
        return detector.image_buffer / res.intensity_at(1).max()

    commands_0 = 2 * (torch.rand(n_modes) - 0.5) * 0.1
    dm._commands = commands_0

    ncpa_dm._commands = 2 * (torch.rand(100) - 0.5) * 0.1
    ncpa_dm.commands[:3] = 0.0

    res = system.run(source=source)
    print(*[(i, el) for i, el in enumerate(system.elements)], sep="\n")
    print(res.intensity_at(1).max())

    res.field_at(5).plot()
    plt.draw()

    res.field_at(6).plot()
    plt.draw()

    plt.figure()
    plt.imshow(response_function(res))
    plt.colorbar()
    plt.draw()

    observed_image = response_function(res)
    detector.photon_noise = False
    detector.readout_noise_variance = 0

    # coeffs = torch.nn.Parameter(
    #     torch.clone(commands_0) + 0.01 * torch.rand(dm.commands.shape)
    # )
    coeffs = torch.nn.Parameter(0.1 * torch.rand(dm.commands.shape, requires_grad=True))
    optimizer = torch.optim.Adam([coeffs], lr=1e-2)
    # optimizer = torch.optim.SGD([coeffs], lr=1e9)

    loss_plot = []

    n_iter = 100

    loss_fig, loss_ax = plt.subplots()

    for i in range(n_iter):

        optimizer.zero_grad()

        simulated_image = forward_model(
            system, source, commands=coeffs, response_function=response_function
        )

        # plt.imshow(simulated_image.detach())
        # plt.draw()
        # plt.pause(1e-6)

        sim = simulated_image
        obs = observed_image

        sim = torch.clamp(sim, min=1e-8)

        loss = torch.sum((simulated_image - observed_image) ** 2)

        loss.backward()

        optimizer.step()
        loss_plot += [loss.item()]

        print(i, f"{loss.item():.5e}", end="\r")

    plt.figure()
    plt.plot(commands_0)
    plt.plot(coeffs.detach())
    plt.draw()

    loss_ax.plot(loss_plot)
    loss_ax.set_xlim(0, n_iter)
    loss_ax.set_yscale("log")
    plt.draw()

    fig, ax = plt.subplots(1, 3)
    fig.suptitle("$I_{fit} - I_{true}$")
    ax[0].imshow((simulated_image).detach())
    ax[1].imshow((observed_image).detach())
    ax[2].imshow(observed_image.detach() - simulated_image.detach())
    plt.draw()

    fig, ax = plt.subplots(1, 3)
    fig.suptitle("Low order DM commands")
    dm._commands = torch.nn.Parameter(commands_0)
    A = dm.opd.detach()
    ax[0].imshow(A)
    dm._commands = coeffs
    B = dm.opd.detach()
    ax[1].imshow(B)
    ax[2].imshow(B - A)
    plt.draw()

    plt.show()
