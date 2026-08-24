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
    Step,
)
from fiatlux.utils.converter import FocalPlaneConverter
from fiatlux.utils.fits_loader import load_pupil, load_zelda_measurement
from fiatlux.optics.elements.field_stop import ShanonFieldStop
from fiatlux.system.optical_system import SerialSystem
from fiatlux.optics.detector import Detector
from fiatlux.optics.elements.deformable_mirror import (
    DeformableMirror,
    ActuatorGrid,
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

import imageio

if __name__ == "__main__":
    # %% DEFINITIONS
    # Define source spectrum
    spectrum = Spectrum.from_sampling(magnitude=5, band=PhotometricBand.HCM, Nu=512)

    # System parameters
    f = 0.125
    lambda_max = spectrum.wavelengths.max()
    D = 38.542

    pupil_mask, (x0, y0, r) = load_pupil("/Users/janinpop/Downloads/outputs/pupil.fits")

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

    step = Step(grid=pupil_grid, piston=lambda_max / 4)

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
            step,
            mft0,
            zelda_stop,
            mft1,
            empty_propagator1,
        ],
    )

    def response_function(result):
        detector.acquire(
            (result.field_at(step) - result.field_at(empty_propagator1))
            + torch.exp(1j * torch.as_tensor(torch.pi) / 2)
            * result.field_at(empty_propagator1)
        )
        return detector.image_buffer

    res = system.run(source=source)
    cube = load_zelda_measurement(
        "/Users/janinpop/Downloads/outputs/differential_piston.fits"
    )

    from matplotlib.patches import Rectangle

    p_simu = []
    p_bench = []

    filenames = []

    matlab_colors = [
        (0.0000, 0.4470, 0.7410),  # blue
        (0.8500, 0.3250, 0.0980),  # orange
        (0.9290, 0.6940, 0.1250),  # yellow
        (0.4940, 0.1840, 0.5560),  # purple
        (0.4660, 0.6740, 0.1880),  # green
        (0.3010, 0.7450, 0.9330),  # light blue
        (0.6350, 0.0780, 0.1840),  # dark red
    ]
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=matlab_colors)

    for i, opd in enumerate(torch.linspace(0, lambda_max, 256)):
        step.piston = opd
        step.build(spectrum=spectrum)
        res = system.run(source=source)

        image = response_function(res)
        image_bench = cube[i, y0 - r : y0 + r, x0 - r : x0 + r] * pupil_mask

        r0 = Rectangle(
            (40, 50), 25, 25, edgecolor=matlab_colors[0], facecolor="none", linewidth=2
        )
        r1 = Rectangle(
            (40, 50), 25, 25, edgecolor=matlab_colors[0], facecolor="none", linewidth=2
        )
        r2 = Rectangle(
            (150, 50), 25, 25, edgecolor=matlab_colors[1], facecolor="none", linewidth=2
        )
        r3 = Rectangle(
            (150, 50), 25, 25, edgecolor=matlab_colors[1], facecolor="none", linewidth=2
        )

        fig_sim, ax_sim = plt.subplots()
        fig_sim.suptitle(f"Piston value : {opd/lambda_max:.2E} $\lambda$")
        ax_sim.imshow(image)
        ax_sim.add_patch(r0)
        ax_sim.add_patch(r2)
        fname = f"./outputs/animations/frame_sim_{i}.png"
        plt.savefig(fname)
        filenames.append(fname)
        plt.close()

        fig_bench, ax_bench = plt.subplots()
        fig_bench.suptitle(f"Piston bit value : {i}")
        ax_bench.imshow(image_bench)
        ax_bench.add_patch(r1)
        ax_bench.add_patch(r3)
        fname = f"./outputs/animations/frame_bench_{i}.png"
        plt.savefig(fname)
        filenames.append(fname)
        plt.close()

        p_simu += [[image[50:75, 50:75].sum(), image[50:75, 150:175].sum()]]
        p_bench += [
            [image_bench[50:75, 50:75].sum(), image_bench[50:75, 150:175].sum()]
        ]


import numpy as np

p_simu = np.vstack(p_simu)
p_simu /= p_simu.max()
p_bench = np.vstack(p_bench)
p_bench /= p_bench.max()


plt.figure()
plt.plot(torch.linspace(0, lambda_max, 256), p_simu)
plt.plot(torch.linspace(0, lambda_max, 256), p_simu[:, 0] - p_simu[:, 1])
plt.xlabel("$\Delta p$ (in nm)")
plt.legend(["Left square", "Right square", "Difference"])
plt.draw()
plt.savefig("./outputs/piston_step_sim.png")

plt.figure()
plt.plot(torch.linspace(0, 255, 256), p_bench)
plt.plot(torch.linspace(0, 255, 256), p_bench[:, 0] - p_bench[:, 1])
plt.xlabel("$\Delta p$ (in bit)")
plt.legend(["Left square", "Right square", "Difference"])
plt.draw()
plt.savefig("./outputs/piston_step_bench.png")
plt.show()
