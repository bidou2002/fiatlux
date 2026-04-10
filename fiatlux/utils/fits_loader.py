from astropy.io import fits


def load_pupil():
    with fits.open("/Users/pjanin/Downloads/pupil.fits") as hdul:
        y0 = int(hdul[0].header["Y0"])
        x0 = int(hdul[0].header["X0"])
        r = int(hdul[0].header["R"])
        pupil = hdul[0].data[y0 - r : y0 + r, x0 - r : x0 + r]
    return pupil


def load_zelda_measurement():
    with fits.open("/Users/pjanin/Downloads/tip0.fits") as hdul:
        y0 = int(hdul[0].header["Y0"])
        x0 = int(hdul[0].header["X0"])
        r = int(hdul[0].header["R"])
        return (
            hdul[0].data[1, y0 - r : y0 + r, x0 - r : x0 + r]
            - hdul[0].data[0, y0 - r : y0 + r, x0 - r : x0 + r]
        )
