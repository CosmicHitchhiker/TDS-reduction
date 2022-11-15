#! /usr/bin/python3

import numpy as np
import geometry as gm
from matplotlib import pyplot as plt
import bias
import dark
import cosmics
import corrections
import dispersion
import flat
from astropy.io import fits
import argparse
from genfuncs import open_fits_array_data


def get_distorsion_map(data, verbose=True):
    h = 10
    d = 3

    y, x = np.shape(data)
    y = np.arange(y)
    x = np.arange(x)

    data_T = data.T

    # За fwhm считаем fwhm (в пикселях) среднего (по X) столбца
    fwhm = gm.calc_fwhm(data[int(len(data_T) / 2)])
    print(('fwhm = ', fwhm, 'pix\n') if verbose else '', end='')

    # Пики в каждом столбце (list из ndarray разной длины)
    def find_peaks_(row):
        return(gm.find_peaks(row, fwhm=fwhm, h=h, d=d))
    peaks = list(map(find_peaks_, data_T))
    print('***all peaks are found***' if verbose else '')

    peaks, n_lines = gm.find_lines_cluster(peaks, y, verbose=True)
    peaks = gm.refine_peaks_i(data_T, peaks, fwhm)
    return peaks


def get_distorsion_file(data, bias_obj=None, dark_obj=None, flat_obj=None,
                        cosm_obj=None, wl_obj=None):
    data_copy = {'data': data.copy()}
    data_copy = bias.process_bias(data_copy, bias_obj)
    data_copy = flat.process_flat(data_copy, flat_obj)
    if cosm_obj:
        data_copy = cosmics.process_cosmics(data_copy)
    data_copy = dispersion.process_dispersion(data_copy, wl_obj)

    dist_calib = np.sum(data_copy['data'], axis=0)
    corr_map, new_y, mean_peaks = corrections.get_correction_map(dist_calib.T,
                                    verbose=True, h=3, d=3.5, return_peaks=True)
    print(mean_peaks)
    middle_peak = mean_peaks[np.argmin(np.abs(mean_peaks - 256))]
    dist_map = corr_map.T
    res = fits.PrimaryHDU(dist_map)
    res.header['GOODY1'] = np.min(new_y)
    res.header['GOODY2'] = np.max(new_y)
    res.header['CRPIX2'] = middle_peak
    res.header['CRPIX2A'] = np.min(mean_peaks)
    return res


def distorsion_from_file(corrections_file):
    if isinstance(corrections_file, str):
        corrections_file = fits.open(corrections_file)[0]

    y1 = corrections_file.header['GOODY1']
    y2 = corrections_file.header['GOODY2'] + 1
    ref_y = corrections_file.header['CRPIX2']
    ref_y2 = corrections_file.header['CRPIX2A']
    good_y = np.arange(y1, y2)
    res = {'data': corrections_file.data, 'new_y': good_y,
           'ref_y': [ref_y, ref_y2]}
    return res


def process_distorsion(data, corr_obj):
    data_copy = data.copy()
    if corr_obj is None:
        return data_copy
    new_y = corr_obj['new_y']
    corr_map = (corr_obj['data'].T)[:, new_y]
    corrected = [(corrections.interpolate_correction_map(x.T, corr_map)).T
                 for x in data_copy['data']]
    data_copy['data'] = np.array(corrected)

    if 'errors' in data_copy:
        errors = [(corrections.interpolate_correction_map(x.T, corr_map)).T
                  for x in data_copy['errors']]
        data_copy['errors'] = np.array(errors)
    if 'mask' in data_copy:
        mask = [(corrections.interpolate_correction_map(x.T, corr_map)).T
                for x in data_copy['mask']]
        data_copy['mask'] = np.array(mask).astype('bool')
    if 'keys' in data_copy:
        data_copy['keys']['CRPIX2'] = corr_obj['ref_y'][0]
        data_copy['keys']['CRPIX2A'] = corr_obj['ref_y'][1]
        data_copy['keys']['CRVAL2'] = 0
        data_copy['keys']['CDELT2'] = 3.7
    return data_copy


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+',
                        help="fits files with stars for distorsion correction")
    parser.add_argument('-d', '--dir', help="directory with input files")
    parser.add_argument('-o', '--out', default='../data/distorsion_map.fits',
                        help='output file')
    parser.add_argument('-B', '--BIAS', help="bias frame (fits) to substract")
    parser.add_argument('-D', '--DARK',
                        help="prepared fits-file with dark frames")
    parser.add_argument('-C', '--COSMICS', action='store_true',
                        help="set this argument to clear cosmic hints")
    parser.add_argument('-W', '--WAVELENGTHS',
                        help="wavelength calibration map")
    parser.add_argument('-F', '--FLAT',
                        help="prepared fits-file with flat frame")
    pargs = parser.parse_args(args[1:])

    if pargs.BIAS:
        bias_obj = bias.bias_from_file(pargs.BIAS)
    else:
        bias_obj = None

    if pargs.DARK:
        dark_obj = dark.dark_from_file(pargs.DARK)
    else:
        dark_obj = None

    if pargs.COSMICS:
        if_clear_cosmics = True
    else:
        if_clear_cosmics = False

    if pargs.WAVELENGTHS:
        wl_obj = dispersion.dispersion_from_file(pargs.WAVELENGTHS)
    else:
        wl_obj = None

    if pargs.FLAT:
        flat_obj = flat.flat_from_file(pargs.FLAT)
    else:
        flat_obj = None

    stars_names = pargs.filenames
    if pargs.dir:
        stars_names = [pargs.dir + x for x in stars_names]
    stars_files, headers = open_fits_array_data(stars_names, header=True)

    corr_file = get_distorsion_file(stars_files, bias_obj, dark_obj,
                                    flat_obj, if_clear_cosmics, wl_obj)
    dist_obj = distorsion_from_file(corr_file)
    data = {'data': stars_files}
    res = process_distorsion(data, dist_obj)
    plt.imshow(np.sum(res['data'], axis=0))
    plt.show()
    corr_file.writeto(pargs.out, overwrite=True)
    return(0)


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
