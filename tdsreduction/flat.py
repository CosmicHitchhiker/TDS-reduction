#! /usr/bin/python3

import numpy as np
import bias
import dark
import corrections
from astropy.io import fits
import scipy.interpolate as interp
import argparse
from genfuncs import open_fits_array_data
from matplotlib import pyplot as plt


def get_spline_setup(flat_string, x=None, band='R'):

    if x is None:
        x = np.arange(len(flat_string))

    if band == 'R':
        prec = 5.
    elif band == 'B':
        prec = 1.5
    else:
        prec = 3.

    need_to_change = 'Yes'
    while need_to_change != '':
        spl = interp.UnivariateSpline(x, flat_string, s=prec*1e+8)
        plt.plot(x, flat_string)
        plt.plot(x, spl(x))
        plt.show()

        params = argparse.ArgumentParser(exit_on_error=False)
        params.add_argument('-p', type=float, default=prec)
        parags = params.parse_args('')
        print(parags)
        need_to_change = input("Change any parameters?(leave blank if No)")
        if need_to_change:
            parags = params.parse_args(need_to_change.split())
            prec = parags.p
    return spl(x)


def get_flat_file(flat_frames, flat_headers,
                  bias_obj=None, dark_obj=None, corr_obj=None):
    verbose = True
    # Половина количества линий из середины для усреднения
    dy = 10

    data = {'data': flat_frames.copy()}
    data = bias.process_bias(data, bias_obj)

    flat = np.sum(data['data'], axis=0)

    x = np.arange(len(flat[0]))
    # Находим средний ряд
    mean = int(len(flat) / 2.)
    flat_string = np.mean(flat[mean - dy:mean + dy], axis=0)

    # if flat_headers[0]['DISP'] == 'R':
    #     spl = interp.UnivariateSpline(x, flat_string, s=5e+8)
    # elif flat_headers[0]['DISP'] == 'B':
    #     spl = interp.UnivariateSpline(x, flat_string, s=1.5e+8)
    # else:
    #     raise ValueError

    flat_string_s = get_spline_setup(flat_string, x, flat_headers[0]['DISP'])
    # if verbose:
    #     plt.figure()
    #     plt.plot(x, flat_string)
    #     plt.plot(x, flat_string_s)
    #     plt.show()
    theor_flat = np.ones(len(flat))[:, np.newaxis] @ flat_string_s[np.newaxis]

    theor_flat = corrections.interpolate_correction_map(theor_flat,
                                                        corr_obj['data'],
                                                        inverse=True)
    flat_coeff = theor_flat / flat
    res = fits.PrimaryHDU(flat_coeff, header=flat_headers[0])

    return res


def flat_from_file(flat_file):
    if isinstance(flat_file, str):
        flat_file = fits.open(flat_file)[0]

    res = {'data': flat_file.data}
    return res


def process_flat(data, flat_obj):
    data_copy = data.copy()
    if flat_obj is None:
        return data_copy
    flat_mask = (flat_obj['data'] > 5)
    data_copy['data'] = data_copy['data'] * flat_obj['data']
    if 'errors' in data_copy:
        data_copy['errors'] = data_copy['errors'] * flat_obj['data']
    if 'mask' in data_copy:
        data_copy['mask'] = data_copy['mask'] | flat_mask
    return data_copy


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+',
                        help="fits files with flat field lamp frames")
    parser.add_argument('-d', '--dir', help="directory with input files")
    parser.add_argument('-o', '--out', default='../data/flat.fits',
                        help='output file')
    parser.add_argument('-X', '--GEOMETRY', help="file with correction map")
    parser.add_argument('-B', '--BIAS', help="bias frame (fits) to substract")
    parser.add_argument('-D', '--DARK',
                        help="prepared fits-file with dark frames")
    pargs = parser.parse_args(args[1:])

    if pargs.BIAS:
        bias_obj = bias.bias_from_file(pargs.BIAS)
    else:
        bias_obj = None

    if pargs.DARK:
        dark_obj = dark.dark_from_file(pargs.DARK)
    else:
        dark_obj = None

    if pargs.GEOMETRY:
        corr_obj = corrections.corrections_from_file(pargs.GEOMETRY)
    else:
        corr_obj = None

    flat_names = pargs.filenames
    if pargs.dir:
        flat_names = [pargs.dir + x for x in flat_names]
    flat_files, headers = open_fits_array_data(flat_names, header=True)

    flat_file = get_flat_file(flat_files, headers, bias_obj,
                              dark_obj, corr_obj)
    flat_file.writeto(pargs.out, overwrite=True)
    return(0)


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
