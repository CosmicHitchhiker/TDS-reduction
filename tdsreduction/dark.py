#! /usr/bin/python3
"""This module works with 'dark' calibration files"""

import numpy as np
import bias
from scipy.interpolate import interp1d
from astropy.io import fits


def get_dark_interp(darks):
    # На входе - словарь экспозиция-дарккадр
    dark_t = np.array(list(darks.keys()))
    dark_img = np.array(list(darks.values()))
    dark = interp1d(dark_t, dark_img, axis=0)
    return dark


def get_dark_file(data, headers, bias_obj=None):
    times = [x["EXPOSURE"] for x in headers]

    dark_times = sorted(set(times))
    times = np.array(times)
    dark_frames = fits.HDUList()
    for dark_exp in dark_times:
        darks_t = data[times == dark_exp]
        # bias.get_bias is sigma-clipped mean
        dark, _ = bias.get_bias(darks_t)
        pre_dark_obj = {'data': dark, 'errors': None}
        dark = (bias.process_bias(pre_dark_obj, bias_obj))['data']
        dark[dark < 0] = 0

        # first header with dark_exp exposition
        hdr = [headers[i] for i in range(len(headers))
               if times[i] == dark_exp][0]
        dark_frames.append(fits.ImageHDU(dark, header=hdr))

    return dark_frames


def dark_from_file(dark_file):
    if isinstance(dark_file, str):
        dark_file = fits.open(dark_file)
    darks = {x.header["EXPOSURE"]: x.data for x in dark_file}
    dark = get_dark_interp(darks)
    return dark


def process_dark(data, dark=None, exposures=None):
    if dark is None:
        return data
    data_res = [frame - dark(t) for frame, t in zip(data['data'], exposures)]
    return {'data': data_res, 'errors': data['errors']}


def main(args=None):
    """This method runs if the file is running as a program"""
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+',
                        help="fits files with dark frames")
    parser.add_argument('-d', '--dir', help="directory with input files")
    parser.add_argument('-o', '--out', default='../data/dark.fits',
                        help='output file')
    parser.add_argument('-B', '--BIAS', help="bias frame (fits) to substract")
    pargs = parser.parse_args(args[1:])

    if pargs.BIAS:
        bias_obj = bias.bias_from_file(pargs.BIAS)
    else:
        bias_obj = None

    dark_names = pargs.filenames
    if pargs.dir:
        dark_names = [pargs.dir + x for x in dark_names]
    dark_files, headers = open_fits_array_data(dark_names, header=True)
    # print(headers[0])

    hdul = get_dark_file(dark_files, headers, bias_obj)
    hdul.writeto(pargs.out, overwrite=True)

    return 0


if __name__ == '__main__':
    import sys
    from utils import open_fits_array_data
    import argparse
    sys.exit(main(sys.argv))
