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
    spectrum = Spectrum.from_sampling(magnitude=9, band=PhotometricBand.HCM, Nu=512)

    # System parameters
    f = 0.125
    lambda_max = spectrum.wavelengths.max()
    D = 38.542

    pupil_mask, (x0, y0, r) = load_pupil("/Users/janinpop/Downloads/outputs/pupil.fits")
    pupil_mask = torch.as_tensor(pupil_mask)

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
    aperture = ArbitraryAperture(
        grid=pupil_grid, transmission=torch.as_tensor(pupil_mask)
    )

    adc_model = ADCDispersionModel()
    amplitudes = adc_model.get_dispersion(angle=30, spectrum=spectrum)
    print(amplitudes / (lambda_max / D))
    adc = ADC(grid=pupil_grid, amplitude=amplitudes, angle=90)

    tip_tilt_compensator = TipTilt(grid=pupil_grid, tip=0, tilt=-amplitudes.max() / 2)

    n_modes = 2
    basis = ZernikeBasis(pixel_grid=pupil_grid, n=n_modes)

    # DM
    dm = DeformableMirror(
        grid=pupil_grid,
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
    zelda_stop = ZeldaStop(grid=focal_grid, radius=1.06 * f * lambda_max / D)

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
            (result.field_at(dm) - result.field_at(empty_propagator1))
            + torch.exp(1j * torch.as_tensor(torch.pi) / 2)
            * result.field_at(empty_propagator1)
        )
        return detector.image_buffer

    res = system.run(source=source)
    reference_image = response_function(res)

    # %% LOAD IMAGE
    cube = load_zelda_measurement("/Users/janinpop/Downloads/outputs/tilt_motor.fits")
    # %% FIT
    torch.manual_seed(0)

    # INIT DM COMMANDS
    commands_0 = 2 * (torch.rand(n_modes) - 0.5) * 0.0
    dm._commands = commands_0

    res = system.run(source=source)
    reference_image = response_function(res)
    print(*[(i, el) for i, el in enumerate(system.elements)], sep="\n")

    import numpy as np

    COEFFS = []
    fig_coeffs, ax_coeffs = plt.subplots()
    fig_coeffs_history, ax_coeffs_history = plt.subplots()
    fig_im, ax_im = plt.subplots(1, 3)

    for i in range(50, 51):

        observed_image = cube[i, y0 - r : y0 + r, x0 - r : x0 + r]
        observed_image = (
            torch.tensor(observed_image.astype(observed_image.dtype.newbyteorder("=")))
            * pupil_mask
        )

        coeffs = torch.nn.Parameter(
            0.1 * torch.rand(dm.commands.shape, requires_grad=True)
        )
        coeffs = torch.nn.Parameter(torch.tensor([0.0, 0.0], requires_grad=True))
        optimizer = torch.optim.Adam([coeffs], lr=1e-1)

        n_iter = 200
        history = []

        for i in range(n_iter):

            optimizer.zero_grad()

            simulated_image = (
                (
                    forward_model(
                        system,
                        source,
                        commands=coeffs,
                        response_function=response_function,
                    )
                )
                * pupil_mask
                * 0.001567428140352112
            )

            loss = torch.sum((simulated_image - observed_image) ** 2)

            loss.backward()

            optimizer.step()
            print(i, f"{loss.item():.5e}", end="\r")

            history.append(coeffs.detach().cpu().numpy().copy())

        sim_sum = simulated_image.sum().item()
        obs_sum = observed_image.sum().item()

        print("Flux ratio:", obs_sum / sim_sum)

        COEFFS.append([coeffs.detach().numpy()])
        COEFFS_stack = np.vstack(COEFFS)

        ax_coeffs_history.clear()
        pcm = ax_coeffs_history.plot(np.vstack(history))
        ax_coeffs.legend(["Tip (x)", "Tilt (y)", "Focus (z)"])
        plt.draw()

        ax_coeffs.clear()
        pcm = ax_coeffs.plot(COEFFS_stack)
        ax_coeffs.legend(["Tip (x)", "Tilt (y)", "Focus (z)"])
        plt.draw()

        fig_im.suptitle("$I_{fit} - I_{true}$")
        ax_im[0].imshow((observed_image).detach())
        ax_im[0].set_title("Bench image")
        ax_im[1].imshow((simulated_image).detach())
        ax_im[1].set_title("Simulated image")
        ax_im[2].imshow(
            (simulated_image.detach() - observed_image.detach())
            / observed_image.detach()
        )
        ax_im[2].set_title("Normalized difference")
        plt.draw()
        plt.pause(0.01)

    plt.show()
