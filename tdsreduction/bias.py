#! /usr/bin/python3

import numpy as np
import astropy.stats as astats


def get_bias(bias, gain=1):
    '''Calculate superbias and readnoise.

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
    '''
    bias_clean = astats.sigma_clip(bias, sigma=5)
    read_noise = np.median(astats.mad_std(bias, axis=(1, 2))) * gain
    superbias = np.average(bias_clean, axis=0)
    superbias = superbias.filled(superbias.mean())
    return (superbias, read_noise)


def get_bias_file(bias_array, header=None):
    superbias, readnoise = get_bias(bias_array)
    bias_file = fits.PrimaryHDU(superbias.astype('float32'), header=header)
    bias_file.header['READNOIS'] = readnoise
    return(bias_file)


def bias_from_file(bias_file):
    bias = bias_file.data
    readnoise = bias_file.header['READNOIS']
    return(bias, readnoise)


def process_bias(data, bias):
    data = data - bias
    return(data)


def main(args=None):
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
    return(0)


if __name__ == '__main__':
    import sys
    from utils import open_fits_array_data
    from astropy.io import fits
    import argparse
    sys.exit(main(sys.argv))
