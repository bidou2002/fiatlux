from copy import deepcopy
import pickle

import pytest
import torch

from fiatlux.core.grid import Grid
from fiatlux.optics.atmosphere import (
    KolmogorovAtmosphereModel,
    NCPAModel,
)
from fiatlux.optics.detector import Detector
from fiatlux.optics.elements.mask import Atmosphere, NCPA, Random


def make_grid(device="cpu"):
    return Grid(
        nx=12,
        ny=10,
        dx=0.1,
        dy=0.12,
        device=torch.device(device),
    )


def test_random_masks_with_the_same_seed_replay_the_same_sequence():
    first = Random(make_grid(), amplitude=100e-9, seed=42)
    second = Random(make_grid(), amplitude=100e-9, seed=42)

    for _ in range(3):
        first._build_opd()
        second._build_opd()
        torch.testing.assert_close(first.opd, second.opd)


def test_different_random_mask_seeds_produce_different_screens():
    first = Random(make_grid(), amplitude=100e-9, seed=1)
    second = Random(make_grid(), amplitude=100e-9, seed=2)

    first._build_opd()
    second._build_opd()

    assert not torch.equal(first.opd, second.opd)


def test_running_one_element_does_not_advance_another_element():
    first = Random(make_grid(), amplitude=1.0, seed=7)
    second = Random(make_grid(), amplitude=1.0, seed=7)

    first._build_opd()
    expected_first_screen = first.opd.clone()
    first._build_opd()
    second._build_opd()

    torch.testing.assert_close(second.opd, expected_first_screen)


def test_stochastic_elements_do_not_modify_the_global_generator():
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)

    random_mask = Random(make_grid(), amplitude=1.0, seed=9)
    random_mask._build_opd()

    torch.testing.assert_close(torch.rand(4), expected)


def test_generator_state_can_be_checkpointed_and_restored():
    model = NCPAModel(make_grid(), opd_rms=100e-9, seed=12)

    state = model.get_rng_state()
    expected = model.sample_opd()
    model.sample_opd()
    model.set_rng_state(state)

    torch.testing.assert_close(model.sample_opd(), expected)


def test_deepcopy_preserves_the_current_generator_state():
    model = KolmogorovAtmosphereModel(make_grid(), r0=0.2, seed=4)
    model.sample_opd()
    copied = deepcopy(model)

    torch.testing.assert_close(model.sample_opd(), copied.sample_opd())


def test_pickle_preserves_the_current_generator_state():
    random_mask = Random(make_grid(), amplitude=1.0, seed=14)
    random_mask._build_opd()
    restored = pickle.loads(pickle.dumps(random_mask))

    random_mask._build_opd()
    restored._build_opd()

    torch.testing.assert_close(random_mask.opd, restored.opd)


def test_wrappers_accept_independent_seeds():
    grid = make_grid()
    atmosphere_model = KolmogorovAtmosphereModel(grid, r0=0.2, seed=1)
    ncpa_model = NCPAModel(grid, opd_rms=100e-9, seed=1)

    atmosphere_a = Atmosphere(grid, atmosphere_model, seed=31)
    atmosphere_b = Atmosphere(grid, atmosphere_model, seed=31)
    ncpa_a = NCPA(grid, ncpa_model, seed=47)
    ncpa_b = NCPA(grid, ncpa_model, seed=47)

    atmosphere_a._build_opd()
    atmosphere_b._build_opd()
    ncpa_a._build_opd()
    ncpa_b._build_opd()

    torch.testing.assert_close(atmosphere_a.opd, atmosphere_b.opd)
    torch.testing.assert_close(ncpa_a.opd, ncpa_b.opd)


def test_detector_accepts_an_external_generator():
    first_generator = torch.Generator().manual_seed(8)
    second_generator = torch.Generator().manual_seed(8)
    first = Detector(make_grid(), photon_noise=True, generator=first_generator)
    second = Detector(make_grid(), photon_noise=True, generator=second_generator)
    expected = torch.full((10, 12), 20.0)

    torch.testing.assert_close(
        first.add_photon_noise(expected),
        second.add_photon_noise(expected),
    )


def test_seed_and_generator_are_mutually_exclusive():
    generator = torch.Generator().manual_seed(1)

    with pytest.raises(ValueError, match="either seed or generator"):
        Random(make_grid(), amplitude=1.0, seed=1, generator=generator)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_reproducibility_on_cuda():
    first = Random(make_grid("cuda"), amplitude=1.0, seed=22)
    second = Random(make_grid("cuda"), amplitude=1.0, seed=22)

    first._build_opd()
    second._build_opd()

    assert first.opd.is_cuda
    torch.testing.assert_close(first.opd, second.opd)
