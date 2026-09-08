import pytest
import torch

from fiatlux.core.spectrum import Band, Spectrum


MONOCHROMATIC_BAND = Band(
    central_wavelength=1.2e-6,
    delta_wavelength=0.0,
    f0=1.1e12,
)


def test_monochromatic_spectrum_has_one_central_finite_channel():
    spectrum = Spectrum(magnitude=5, band=MONOCHROMATIC_BAND, samples=1)

    assert spectrum.wavelengths.shape == (1,)
    assert spectrum.fluxes.shape == (1,)
    torch.testing.assert_close(
        spectrum.wavelengths,
        torch.tensor([MONOCHROMATIC_BAND.central_wavelength]),
    )
    assert torch.isfinite(spectrum.fluxes).all()
    assert spectrum.fluxes.item() == pytest.approx(
        MONOCHROMATIC_BAND.photon_flux(5)
    )


@pytest.mark.parametrize(
    "band",
    [
        MONOCHROMATIC_BAND,
        Band(central_wavelength=1.2e-6, delta_wavelength=1e-12, f0=1.1e12),
    ],
)
def test_from_sampling_guarantees_at_least_one_channel(band):
    spectrum = Spectrum.from_sampling(magnitude=5, band=band, Nu=64)

    assert spectrum.wavelengths.shape == (1,)
    assert spectrum.fluxes.shape == (1,)
    assert spectrum.wavelengths.item() == pytest.approx(band.central_wavelength)
    assert torch.isfinite(spectrum.fluxes).all()


def test_all_constructors_return_the_same_object_structure():
    direct = Spectrum(magnitude=5, band=MONOCHROMATIC_BAND, samples=1)
    sampled = Spectrum.from_sampling(magnitude=5, band=MONOCHROMATIC_BAND, Nu=64)

    assert vars(direct).keys() == vars(sampled).keys()
    assert sampled.magnitude == direct.magnitude
    torch.testing.assert_close(sampled.wavelengths, direct.wavelengths)
    torch.testing.assert_close(sampled.fluxes, direct.fluxes)


def test_from_sampling_preserves_requested_multichannel_spacing_and_total_flux():
    band = Band(central_wavelength=1.1e-6, delta_wavelength=0.2e-6, f0=1e12)
    spectrum = Spectrum.from_sampling(magnitude=3, band=band, Nu=128)
    expected_spacing = 2 * (
        band.central_wavelength + band.delta_wavelength / 2
    ) / 128

    assert len(spectrum.wavelengths) > 1
    torch.testing.assert_close(
        torch.diff(spectrum.wavelengths),
        torch.full_like(torch.diff(spectrum.wavelengths), expected_spacing),
    )
    assert spectrum.fluxes.sum().item() == pytest.approx(
        band.photon_flux(3), rel=1e-6
    )


@pytest.mark.parametrize("samples", [0, -1, 1.5, True])
def test_direct_constructor_rejects_invalid_sample_counts(samples):
    with pytest.raises(ValueError, match="samples must be a positive integer"):
        Spectrum(magnitude=0, band=MONOCHROMATIC_BAND, samples=samples)


@pytest.mark.parametrize("Nu", [0, -1, 1.5, True])
def test_from_sampling_rejects_invalid_grid_sizes(Nu):
    with pytest.raises(ValueError, match="Nu must be a positive integer"):
        Spectrum.from_sampling(magnitude=0, band=MONOCHROMATIC_BAND, Nu=Nu)
