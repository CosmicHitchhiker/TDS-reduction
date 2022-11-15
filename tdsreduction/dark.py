#! /usr/bin/python3
"""This module works with 'dark' calibration files"""

import numpy as np
import bias
from scipy.interpolate import interp1d
from astropy.io import fits
import argparse
from genfuncs import open_fits_array_data


def get_dark_interp(darks):
    """Generate interpolation object with dark current for any exposition.

    Generate scipy interp1d object that takes exposition
    and returns dark current in each pixel for that exposition.

    Parameters
    ----------
    darks : dict
        keys - expositions of each dark frames (floats)
        values - dark frames (2D ndarray)

    Returns
    -------
    dark : interp1d (scipy.interpolate.interp1d)
        object takes exposition and reurns dark for it
    """
    # На входе - словарь экспозиция-дарккадр
    dark_t = np.array(list(darks.keys()))
    dark_img = np.array(list(darks.values()))
    dark = interp1d(dark_t, dark_img, axis=0)
    return dark


def get_dark_file(data, headers, bias_obj=None):
    """Prepare dark file with separate hdu for each expopsition.

    Apply sigma-clipping to all given dark images (each exp separately).
    Summarize dark frames for each exposition.
    Subtract bias if given.

    Parameters
    ----------
    data : 3D ndarray
        Array of dark frames.
    headers : list of astropy header
        Headers corresponding to data.
        Headers are used for reading expositions.

    Returns
    -------
    dark_frames : list of fits.ImageHDU
        Resulting file with summarized dark for each exposition.
    """
    times = [x["EXPOSURE"] for x in headers]

    dark_times = sorted(set(times))
    times = np.array(times)
    dark_frames = fits.HDUList()
    for dark_exp in dark_times:
        darks_t = data[times == dark_exp]
        # bias.get_bias is sigma-clipped mean
        dark, _ = bias.get_bias(darks_t)
        pre_dark_obj = {'data': dark}
        dark = (bias.process_bias(pre_dark_obj, bias_obj))['data']
        dark[dark < 0] = 0

        # first header with dark_exp exposition
        hdr = [headers[i] for i in range(len(headers))
               if times[i] == dark_exp][0]
        dark_frames.append(fits.ImageHDU(dark, header=hdr))

    return dark_frames


def dark_from_file(dark_file):
    """Get dark object from prepared dark file.

    Parameters
    ----------
    dark_file : str or list of fits.ImageHDU
        File with array of dark frames or path to it

    Returns
    -------
    dark_obj : scipy.interpolate.interp1d
        object that returns dark for given exposition
    """
    if isinstance(dark_file, str):
        dark_file = fits.open(dark_file)
    darks = {x.header["EXPOSURE"]: x.data for x in dark_file}
    dark = get_dark_interp(darks)
    return dark


def process_dark(data, dark=None, exposures=None):
    """Apply dark calibration to the given data.

    Subtract dark frame of given exposures from all of the given data frames.

    Parameters
    ----------
    data : dict
        'data' - 2D or 3D ndarray, array of data images
        'errors' - 2D or 3D ndarray, array of corresponding errors squared
    dark_obj : scipy.interpolate.interp1d
        object that returns dark for given exposition
    exposures : list of floats
        list of corresponding exposition for each data['data']

    Returns
    -------
    data : dict
        Has the same structure as input data
    """
    if dark is None:
        return data.copy()
    data_copy = data.copy()
    if exposures is None:
        exposures = [h['EXPOSURE'] for h in data_copy['headers']]
    zip_de = zip(data_copy['data'], exposures)
    data_res = np.array([frame - dark(t) for frame, t in zip_de])
    data_copy['data'] = data_res
    if 'mask' in data_copy:
        data_copy['mask'] = data_copy['mask'] | (data_copy['data'] < 0)
    return data_copy


def main(args=None):
    """This method runs if the file is running as a program"""
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+',
                        help="fits files with dark frames")
    parser.add_argument('-d', '--dir', help="directory with input files")
    parser.add_argument('-o', '--out', default='../data/dark.fits',
                        help='output file')
    parser.add_argument('-B', '--BIAS', help="bias frame (fits) to subtract")
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
    sys.exit(main(sys.argv))
