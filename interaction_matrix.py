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
    spectrum = Spectrum.from_sampling(magnitude=18, band=PhotometricBand.HCM2, Nu=512)

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
    aperture = CircularAperture(radius=D / 2)

    adc_model = ADCDispersionModel()
    amplitudes = adc_model.get_dispersion(angle=30, spectrum=spectrum)
    print(amplitudes / (lambda_max / D))
    adc = ADC(amplitude=0 * amplitudes, angle=90)

    tip_tilt_compensator = TipTilt(tip=0, tilt=-amplitudes.max() / 2)

    n_modes = 100

    # actuator_grid = ActuatorGrid(
    #     n_actuators_x=n_modes, n_actuators_y=n_modes, pitch=D / n_modes
    # )

    # basis = GaussianZonalBasis(
    #     actuator_grid=actuator_grid,
    #     pixel_grid=pupil_grid,
    #     influence_width=1,
    # )

    # basis = SquareZonalBasis(
    #     actuator_grid=actuator_grid,
    #     pixel_grid=pupil_grid,
    #     influence_width=D / n_modes,
    # )

    # basis = SquarePTTZonalBasis(
    #     actuator_grid=actuator_grid,
    #     pixel_grid=pupil_grid,
    #     influence_width=D / n_modes,
    # )

    # basis = FourierBasis(pixel_grid=pupil_grid, order=10)

    basis = ZernikeBasis(pixel_grid=pupil_grid, n=n_modes)

    # DM
    dm = DeformableMirror(
        actuator_grid=ActuatorGrid(n_actuators_x=10, n_actuators_y=10, pitch=D / 10),
        pixel_grid=pupil_grid,
        control_basis=basis,
        stroke=1e-6,
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
        source=source,
        detector=detector,
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

    res = system.run()

    plt.figure()
    plt.imshow(torch.sum(res.field_at(6).intensity(), dim=0))
    plt.show()

    im = InteractionMatrix(
        system=system,
        dm=dm,
        poke_amplitude=0.05,
        response_function=lambda result: (result.field_at(5) - result.field_at(-1))
        + torch.exp(1j * torch.as_tensor(torch.pi) / 2) * result.field_at(-1),
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
