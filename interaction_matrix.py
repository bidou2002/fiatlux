from fiatlux.optical_elements.mask import CircularAperture
from fiatlux.grid import Grid
from fiatlux.spectrum import *
from fiatlux.source import *
from fiatlux.propagator import MFTPropagator, IdentityPropagator
from fiatlux.optical_elements.mask import ZeldaMask, ZeldaStop, ADC, TipTilt, Piston
from fiatlux.utils.converter import FocalPlaneConverter
from fiatlux.optical_elements.field_stop import ShanonFieldStop
from fiatlux.optical_system import SerialSystem
from fiatlux.detector import Detector
from fiatlux.optical_elements.deformable_mirror import (
    DeformableMirror,
    ActuatorGrid,
    InteractionMatrix,
    ZonalBasis,
    FourierBasis,
)

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import json
from config.serial_config import SystemConfig

if __name__ == "__main__":

    # with open("./config/zelda.json", "r") as f:
    #     data = json.load(f)

    # # Validate JSON
    # config = SystemConfig(**data)

    # # 1. Spectrum
    # spectrum = Spectrum.from_sampling(
    #     magnitude=config.spectrum.magnitude,
    #     band=getattr(PhotometricBand, config.spectrum.band),
    #     Nu=config.pupil_grid.nx / config.aperture.radius,
    # )
    # lambda_max = spectrum.wavelengths.max()

    # # 2. Grids
    # pupil_grid = Grid(**config.pupil_grid.dict())
    # focal_grid = Grid(**config.focal_grid.dict())

    # # 3. Source
    # source = PlaneWave(spectrum=spectrum)

    # # 4. Identity propagators
    # identity_props = [
    #     IdentityPropagator(grid=pupil_grid) for _ in config.identity_propagators
    # ]

    # # 5. Aperture
    # aperture = CircularAperture(radius=config.aperture.radius)

    # # 6. Actuator grid
    # actuator_grid = ActuatorGrid(**config.actuator_grid.dict())

    # # 7. Basis
    # basis = ZonalBasis(
    #     actuator_grid=ActuatorGrid(**config.basis.actuator_grid.dict()),
    #     pixel_grid=Grid(**config.basis.pixel_grid.dict()),
    #     influence_width=config.basis.influence_width,
    # )

    # # 8. DM
    # dm = DeformableMirror(
    #     actuator_grid=ActuatorGrid(**config.dm.actuator_grid.dict()),
    #     pixel_grid=Grid(**config.dm.pixel_grid.dict()),
    #     control_basis=basis,
    #     stroke=config.dm.stroke,
    # )

    # # 9. MFT propagators
    # mfts = [MFTPropagator(**mft.dict()) for mft in config.mfts]

    # # 10. Zelda mask & stop
    # zelda_mask = ZeldaMask(
    #     radius=config.zelda_mask.radius, well_depth=config.zelda_mask.well_depth
    # )
    # zelda_stop = ZeldaStop(radius=config.zelda_stop.radius)

    # # 11. Detector
    # detector = Detector(**config.detector.dict())

    # # 12. Serial system
    # elements_map = {
    #     "identity0": identity_props[0],
    #     "identity1": identity_props[1],
    #     "aperture": aperture,
    #     "dm": dm,
    #     "mft0": mfts[0],
    #     "mft1": mfts[1],
    #     "zelda_stop": zelda_stop,
    # }
    # elements = [elements_map[name] for name in config.serial_system.elements]

    # system = SerialSystem(source=source, detector=detector, elements=elements)
    # system_res = system.run()

    # # 13. Interaction matrix
    # interaction_matrix = InteractionMatrix(
    #     system=system, dm=dm, poke_amplitude=config.interaction_matrix.poke_amplitude
    # )

    # # -------------------------------
    # # Usage example
    # # -------------------------------
    # # system, res, im = from_json("system_config.json")
    # # print(res)

    # Define source spectrum
    spectrum = Spectrum.from_sampling(magnitude=9, band=PhotometricBand.HCM, Nu=512)

    # System parameters
    f = 0.125
    lambda_max = spectrum.wavelengths.max()
    D = 1.0

    # Pupil plane sampling
    Nx = Ny = 128
    dx = dy = D / Nx
    pupil_grid = Grid(nx=Nx, ny=Ny, dx=dx, dy=dx)

    # Focal plane sampling
    Nu = Nv = 512
    # # Shanon
    # du = dv = Nx * (f * l / D) / Nu
    du = dv = (1) * (f * lambda_max / D) / (Nu / 2)
    focal_grid = Grid(nx=Nu, ny=Nv, dx=du, dy=dv)

    # Create source
    source = PlaneWave(spectrum=spectrum)

    # Empty propagator (for testing)
    empty_propagator0 = IdentityPropagator(pupil_grid)
    empty_propagator1 = IdentityPropagator(pupil_grid)

    # Create aperture
    aperture = CircularAperture(radius=D / 2)

    actuator_grid = ActuatorGrid(n_actuators_x=20, n_actuators_y=20, pitch=D / 20)

    basis = ZonalBasis(
        actuator_grid=actuator_grid,
        pixel_grid=pupil_grid,
        influence_width=D,
    )

    # basis = FourierBasis(pixel_grid=pupil_grid, order=10)

    # DM
    dm = DeformableMirror(
        actuator_grid=ActuatorGrid(n_actuators_x=10, n_actuators_y=10, pitch=D / 10),
        pixel_grid=pupil_grid,
        control_basis=basis,
        stroke=5e-6,
    )

    # Define propagator
    mft0 = MFTPropagator(focal_length=f, output_grid=focal_grid)
    mft1 = MFTPropagator(focal_length=f, output_grid=pupil_grid)

    theta = torch.tensor(lambda_max / 4)
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
        source=source,
        detector=detector,
        elements=[
            empty_propagator0,
            aperture,
            dm,
            mft0,
            zelda_stop,
            mft1,
            empty_propagator1,
        ],
    )
    res = system.run()

    im = InteractionMatrix(
        system=system,
        dm=dm,
        poke_amplitude=0.05,
        response_function=lambda result: (result.field_at(3) - result.field_at(-1))
        + torch.exp(1j * torch.pi / 2 * torch.tensor(1)) * result.field_at(-1),
    )
    im.push_pull(verbose=True)

    U, S, Vh = torch.linalg.svd(im.matrix.to(torch.float32), full_matrices=False)
    plt.figure()
    plt.plot(S)
    plt.yscale("log")
    plt.xlabel("Mode index")
    plt.ylabel("Singular value")
    plt.title("Singular values of the interaction matrix")
    plt.grid()
    plt.draw()

    N = 100
    n = int((N**0.5))

    fig, axes = plt.subplots(n, n, figsize=(10, 10))
    fig_out, axes_out = plt.subplots(n, n, figsize=(10, 10))

    for i in range(N):

        ax = axes[i // n, i % n]
        ax_out = axes_out[i // n, i % n]

        mode = U[:, i].reshape(pupil_grid.nx, pupil_grid.ny)
        mode_in = im.matrix[:, i].reshape(pupil_grid.nx, pupil_grid.ny)

        ax.imshow(mode_in)
        ax.set_title(f"Mode {i}")
        ax.axis("off")

        ax_out.imshow(mode)
        ax_out.set_title(f"Mode {i}")
        ax_out.axis("off")

    plt.tight_layout()
    plt.draw()

    plt.show()
