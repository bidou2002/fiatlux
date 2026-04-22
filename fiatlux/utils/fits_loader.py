from astropy.io import fits
from pathlib import Path


def load_pupil(filepath:Path):
    with fits.open(filepath) as hdul:
        y0 = int(hdul[0].header["Y0"])
        x0 = int(hdul[0].header["X0"])
        r = int(hdul[0].header["R"])
        pupil = hdul[0].data[y0 - r : y0 + r, x0 - r : x0 + r]
    return pupil, (x0, y0, r)


def load_zelda_measurement(filepath:Path):
    with fits.open(filepath) as hdul:
        return hdul[0].data
        
