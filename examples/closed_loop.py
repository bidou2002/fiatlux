from fiatlux import *
import torch
import matplotlib.pyplot as plt
from functools import partial
from fiatlux.utils.fits_loader import load_pupil, load_zelda_measurement


def main():
    # Define source spectrum
    spectrum = Spectrum.from_sampling(magnitude=14, band=PhotometricBand.HCM2, Nu=512)

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
    # Create aperture
    aperture = ArbitraryAperture(
        grid=pupil_grid, transmission=torch.as_tensor(pupil_mask)
    )

    adc_model = ADCDispersionModel()
    amplitudes = adc_model.get_dispersion(angle=30, spectrum=spectrum)
    print(amplitudes / (lambda_max / D))
    adc = ADC(grid=pupil_grid, amplitude=amplitudes, angle=90)

    tip_tilt_compensator = TipTilt(grid=pupil_grid, tip=0, tilt=-amplitudes.max() / 2)

    n_modes = 16

    basis = ZernikeBasis(pixel_grid=pupil_grid, n=n_modes)

    # DM
    dm = DeformableMirror(
        grid=pupil_grid,
        actuator_grid=ActuatorGrid(n_actuators_x=10, n_actuators_y=10, pitch=D / 10),
        pixel_grid=pupil_grid,
        control_basis=basis,
        stroke=1e-6,
    )

    dm_atm = DeformableMirror(
        grid=pupil_grid,
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
    zelda_mask = ZeldaMask(grid=focal_grid, radius=f * lambda_max / D, well_depth=theta)
    zelda_stop = ZeldaStop(grid=focal_grid, radius=f * lambda_max / D)

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
            dm_atm,
            dm,
            mft0,
            zelda_stop,
            mft1,
            empty_propagator1,
        ],
    )

    def acquiring_function(
        system: SerialSystem, source: Source, detector: Detector
    ) -> torch.Tensor:

        result: SimulationResult = system.run(source=source, detector=detector)
        detector.acquire(
            (result.field_at(dm) - result.field_at(empty_propagator1))
            + torch.exp(1j * torch.as_tensor(torch.pi) / 2)
            * result.field_at(empty_propagator1)
        )
        return detector.image_buffer.flatten()  # (nx*ny,)

    interaction_matrix = InteractionMatrix(
        dm=dm,
        poke_amplitude=0.1,
        acquiring_function=partial(
            acquiring_function, system=system, source=source, detector=detector
        ),
    )
    interaction_matrix.push_pull(verbose=True)
    interaction_matrix.compute_control_matrix()

    interaction_matrix.plot()

    torch.manual_seed(1)
    Iref = (
        acquiring_function(system, source, detector)
        .reshape(pupil_grid.nx, pupil_grid.ny)
        .flatten()
    )

    dm_atm._commands = torch.rand_like(dm_atm._commands) * 0.1
    dm._commands = torch.zeros_like(dm._commands)
    gain = 0.5
    fig, ax = plt.subplots(1, 3)

    rms = []

    for iter in range(30):

        rms += [(dm_atm.opd + dm.opd).std()]

        print(f"{iter}", end="\r")
        # Measure current response
        response = acquiring_function(system, source, detector)
        # Compute correction
        correction = interaction_matrix.control_matrix @ (response - Iref)

        # Apply correction
        dm._commands -= gain * correction

        ax[0].imshow(detector.image_buffer)
        ax[1].imshow(dm.opd)
        ax[2].imshow((dm_atm.opd + dm.opd))
        plt.pause(0.1)

    fig, ax = plt.subplots()
    ax.plot(rms)
    ax.set_yscale("log")
    plt.draw()
    plt.show()


if __name__ == "__main__":
    main()
