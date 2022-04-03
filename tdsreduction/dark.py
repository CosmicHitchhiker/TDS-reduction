#! /usr/bin/python3

import numpy as np
from bias import get_bias
from scipy.interpolate import interp1d


def get_dark_interp(darks):
    # На входе - словарь экспозиция-дарккадр
    dark_t = np.array(list(darks.keys()))
    dark_img = np.array(list(darks.values()))
    dark = interp1d(dark_t, dark_img, axis=0)
    return(dark)


def get_dark_k(darks, deg=1):
    t = np.array(list(darks.keys()))
    print("times", t)
    # N, ypix, xpix -> xpix, ypix, N
    # Потому что polyfit не берёт 3D-массивы
    darks = np.array(list(darks.values()))
    darks = darks.T
    print("darks shape", darks.shape)
    # xpix, ypix, deg+1

    def polyf(x):
        np.polyfit(t, x, deg)

    k = np.apply_along_axis(polyf, 2, darks)
    # dark(t) = k[0]*t^deg + k[1]*t^(deg-1) + ... + k[deg]
    k = k.T
    return k


def get_dark_file(data, headers, superbias=0):
    times = [x["EXPOSURE"] for x in headers]

    dark_times = sorted(set(times))
    times = np.array(times)
    HDUs = fits.HDUList()
    for t in dark_times:
        darks_t = data[times == t]
        dark, _ = get_bias(darks_t)
        dark = dark - superbias
        dark[dark < 0] = 0

        hdr = [headers[i] for i in range(len(headers)) if times[i] == t][0]
        HDUs.append(fits.ImageHDU(dark, header=hdr))

    return(HDUs)


def dark_from_file():
    return(0)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+',
                        help="fits files with dark frames")
    parser.add_argument('-d', '--dir', help="directory with input files")
    parser.add_argument('-o', '--out', default='../data/dark.fits',
                        help='output file')
    parser.add_argument('-B', '--BIAS', help="bias frame (fits) to substract")
    pargs = parser.parse_args(args[1:])

    if pargs.BIAS:
        superbias = fits.getdata(pargs.BIAS)
    else:
        superbias = 0

    dark_names = pargs.filenames
    if pargs.dir:
        dark_names = [pargs.dir + x for x in dark_names]
    dark_files, headers = open_fits_array_data(dark_names, header=True)
    # print(headers[0])

    hdul = get_dark_file(dark_files, headers, superbias)
    hdul.writeto(pargs.out, overwrite=True)

    return(0)


if __name__ == '__main__':
    import sys
    from utils import open_fits_array_data
    from astropy.io import fits
    import argparse
    sys.exit(main(sys.argv))
