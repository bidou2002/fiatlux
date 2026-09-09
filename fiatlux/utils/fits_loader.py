from pathlib import Path


def _fits_module():
    try:
        from astropy.io import fits
    except ImportError as error:
        raise ImportError(
            "FITS support requires Astropy; install it with "
            "`python -m pip install 'fiatlux[fits]'`."
        ) from error
    return fits


def load_pupil(filepath:Path):
    fits = _fits_module()
    with fits.open(filepath) as hdul:
        y0 = int(hdul[0].header["Y0"])
        x0 = int(hdul[0].header["X0"])
        r = int(hdul[0].header["R"])
        pupil = hdul[0].data[y0 - r : y0 + r, x0 - r : x0 + r]
    return pupil, (x0, y0, r)


def load_zelda_measurement(filepath:Path):
    fits = _fits_module()
    with fits.open(filepath) as hdul:
        return hdul[0].data
        
