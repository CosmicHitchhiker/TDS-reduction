#! /usr/bin/python3
"""This module works with bias files"""

import numpy as np
import astropy.stats as astats
from astropy.io import fits
import argparse
from genfuncs import open_fits_array_data


def get_bias(bias, gain=1):
    """Calculate superbias and readnoise.

    Apply sigma-clipping to all given bias images.
    Calculate readnoise (median robust standard deviation multiplied by gain)
    Get superbias by averaging all bias images.

    Parameters
    ----------
    bias : 3D ndarray
        Array of bias images.
    gain : float, optional
        Electrons per ADU in given bias images (default is 1).

    Returns
    -------
    suber_bias : 2D ndarray
        Superbias image.
    read_noise : float
        Read noise in the current observations
    """
    bias_clean = astats.sigma_clip(bias, sigma=5)
    read_noise = np.median(astats.mad_std(bias, axis=(1, 2))) * gain
    superbias = np.average(bias_clean, axis=0)
    superbias = superbias.filled(superbias.mean())
    return (superbias, read_noise)


def get_bias_file(bias_array, header=None):
    """Prepare superbias file with readnoise in its header.

    Apply sigma-clipping to all given bias images.
    Calculate readnoise (median robust standard deviation multiplied by gain).
    For TDS gain is assumed to be 1.
    Get superbias by averaging all bias images.

    Parameters
    ----------
    bias : 3D ndarray
        Array of bias images.
    header : astropy header, optional
        This header will be used as a header for the resulting file,
        but new key 'READNOIS' will be added.

    Returns
    -------
    bias_file : fits.PrimaryHDU
        Resulting file with the superbias.
    """
    superbias, readnoise = get_bias(bias_array)
    bias_file = fits.PrimaryHDU(superbias.astype('float32'), header=header)
    bias_file.header['READNOIS'] = readnoise
    return bias_file


def bias_from_file(bias_file):
    """Get supebias frame and readnoise form file.

    Parameters
    ----------
    bias_file : str or fits.PrimaryHDU
        File with superbias frame and 'READNOISE' key in the header.

    Returns
    -------
    bias_obj : dict
        'data' - np.array, superbias frame
        'errors' - np.array, superbias errors squared
        'readnoise' - float, read noise
    """
    if isinstance(bias_file, str):
        bias_file = fits.open(bias_file)[0]
    bias = bias_file.data
    readnoise = bias_file.header['READNOIS']
    errors = (np.ones(np.shape(bias)) * readnoise)**2
    return({'data': bias, 'errors': errors, 'readnoise': readnoise})


def process_bias(data, bias_obj=None):
    """Apply bias calibration to the given data.

    Subtract bias frame from all of the given data frames.

    Parameters
    ----------
    data : dict
        'data' - 2D or 3D ndarray, array of data images
        'errors' - 2D or 3D ndarray, array of corresponding errors squared
    bias_obj : dict
        'data' - np.array, superbias frame
        'errors' - np.array, superbias errors squared
        'readnoise' - float, read noise

    Returns
    -------
    data : dict
        Has the same structure as input data
    """
    data_copy = data.copy()
    if bias_obj is None:
        return data_copy

    data_copy['data'] = data_copy['data'] - bias_obj['data']

    if 'errors' in data_copy:
        data_copy['errors'] = np.sqrt(data_copy['errors']**2
                                      + bias_obj['errors']**2)
    if 'mask' in data_copy:
        data_copy['mask'] = data_copy['mask'] | (data_copy['data'] < 0)
    if 'keys' in data_copy:
        data_copy['keys']['READNOIS'] = bias_obj['readnoise']

    return data_copy


def main(args=None):
    """This method runs if the file is running as a program"""
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+',
                        help="fits files with bias frames")
    parser.add_argument('-d', '--dir', help="directory with input files")
    parser.add_argument('-o', '--out', default='../data/superbias.fits',
                        help='output file')
    pargs = parser.parse_args(args[1:])

    # print(args)
    bias_names = pargs.filenames
    if pargs.dir:
        bias_names = [pargs.dir + x for x in bias_names]
    bias_files, headers = open_fits_array_data(bias_names, header=True)
    superbias_file = get_bias_file(bias_files, headers[0])
    superbias_file.writeto(pargs.out, overwrite=True)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
