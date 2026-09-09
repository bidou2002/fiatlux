import numpy as np
import pytest
import torch

from fiatlux.core.grid import Grid
from fiatlux.optics.elements.mask import (
    HarmoniDatasetNotFoundError,
    HarmoniResiduals,
    InvalidHarmoniDatasetError,
)


fits = pytest.importorskip("astropy.io.fits")


def write_dataset_file(path, values, *, dtype=np.float32):
    data = np.zeros((2, len(values), 2, 3), dtype=dtype)
    data[1] = np.asarray(values, dtype=dtype)[:, None, None]
    fits.PrimaryHDU(data).writeto(path)


def make_grid():
    return Grid(nx=3, ny=2, dx=0.1, dy=0.1)


def test_valid_fits_files_are_sorted_scaled_and_sampled_in_order(tmp_path):
    write_dataset_file(tmp_path / "b.fits", [2.0])
    write_dataset_file(tmp_path / "a.fits", [1.0])

    residuals = HarmoniResiduals(
        make_grid(),
        dataset_path=tmp_path,
        opd_scale=1e-9,
        rotate_quarter_turns=0,
        shuffle=False,
    )

    assert [path.name for path in residuals.files] == ["a.fits", "b.fits"]
    torch.testing.assert_close(
        residuals.datacube[:, 0, 0], torch.tensor([1e-9, 2e-9])
    )
    residuals._build_opd()
    torch.testing.assert_close(residuals.opd, residuals.datacube[0])
    residuals._build_opd()
    torch.testing.assert_close(residuals.opd, residuals.datacube[1])


def test_public_pupil_can_be_authoritative_when_opd_is_zero(tmp_path):
    write_dataset_file(tmp_path / "screens.fits", [0.0])
    pupil = torch.zeros((2, 3), dtype=torch.bool)
    pupil[1, 1] = True

    residuals = HarmoniResiduals(
        make_grid(), tmp_path, pupil=pupil, rotate_quarter_turns=0
    )

    assert residuals.pupil[1, 1]
    assert residuals.support is residuals.pupil


def test_shuffle_order_is_reproducible_for_a_seed(tmp_path):
    write_dataset_file(tmp_path / "screens.fits", [1.0, 2.0, 3.0, 4.0])
    kwargs = dict(dataset_path=tmp_path, rotate_quarter_turns=0, seed=17)
    first = HarmoniResiduals(make_grid(), **kwargs)
    second = HarmoniResiduals(make_grid(), **kwargs)

    first_order = [next(first.iterator) for _ in range(4)]
    second_order = [next(second.iterator) for _ in range(4)]

    assert first_order == second_order
    assert sorted(first_order) == list(range(4))


def test_shuffle_accepts_an_external_generator(tmp_path):
    write_dataset_file(tmp_path / "screens.fits", [1.0, 2.0, 3.0, 4.0])
    first_generator = torch.Generator().manual_seed(29)
    second_generator = torch.Generator().manual_seed(29)

    first = HarmoniResiduals(
        make_grid(),
        tmp_path,
        rotate_quarter_turns=0,
        generator=first_generator,
    )
    second = HarmoniResiduals(
        make_grid(),
        tmp_path,
        rotate_quarter_turns=0,
        generator=second_generator,
    )

    assert [next(first.iterator) for _ in range(4)] == [
        next(second.iterator) for _ in range(4)
    ]


def test_missing_non_directory_and_empty_paths_have_actionable_errors(tmp_path):
    with pytest.raises(HarmoniDatasetNotFoundError, match="does not exist"):
        HarmoniResiduals(make_grid(), tmp_path / "missing")

    file_path = tmp_path / "not-a-directory"
    file_path.write_text("x")
    with pytest.raises(HarmoniDatasetNotFoundError, match="not a directory"):
        HarmoniResiduals(make_grid(), file_path)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(InvalidHarmoniDatasetError, match="contains no FITS"):
        HarmoniResiduals(make_grid(), empty)


@pytest.mark.parametrize(
    "data, message",
    [
        (np.zeros((2, 2, 3), dtype=np.float32), "expected shape"),
        (np.zeros((2, 1, 4, 3), dtype=np.float32), "does not match grid shape"),
        (np.zeros((2, 1, 2, 3), dtype=np.int16), "floating dtype"),
        (np.full((2, 1, 2, 3), np.nan, dtype=np.float32), "NaN or infinite"),
        (np.zeros((2, 0, 2, 3), dtype=np.float32), "contains no screens"),
    ],
)
def test_malformed_fits_data_are_rejected(tmp_path, data, message):
    fits.PrimaryHDU(data).writeto(tmp_path / "bad.fits")

    with pytest.raises(InvalidHarmoniDatasetError, match=message):
        HarmoniResiduals(make_grid(), tmp_path, rotate_quarter_turns=0)


def test_missing_hdu_is_rejected(tmp_path):
    write_dataset_file(tmp_path / "screens.fits", [1.0])

    with pytest.raises(InvalidHarmoniDatasetError, match="missing HDU index 2"):
        HarmoniResiduals(make_grid(), tmp_path, hdu_index=2)
