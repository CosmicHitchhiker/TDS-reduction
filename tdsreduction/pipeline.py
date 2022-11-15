#! /usr/bin/python3

import bias
import flat
import dark
import cosmics
import corrections
import dispersion
import distorsion
import sky
import yaml
import numpy as np
from astropy.io import fits


def my_average(array, weights=None, axis=None):
    if weights is None:
        weights = np.ones_like(array)
    data_copy = array.copy()
    weighted_data = np.sum(data_copy * weights, axis=axis)
    weights_sum = np.sum(weights, axis=axis)
    good_pixels = (weights_sum != 0)
    weighted_data[good_pixels] = weighted_data[good_pixels] \
                                 / weights_sum[good_pixels]
    data_copy = weighted_data
    return data_copy


def fits_from_data(data, summ=False):
    data_copy = data.copy()
    if summ:
        data_copy = sum_data(data_copy)

    hdr = prerpare_header(data_copy)

    result = [fits.PrimaryHDU(data_copy['data'], header=hdr)]
    if 'errors' in data_copy:
        result.append(fits.ImageHDU(data_copy['errors'], name='errors'))
    if 'mask' in data_copy:
        result.append(fits.ImageHDU(data_copy['mask'], name='mask'))

    result = fits.HDUList(result)
    return result


def sum_data(data):
    data_copy = data.copy()
    if 'mask' in data_copy:
        weights = (~data_copy['mask']).astype(int)
        data_copy['mask'] = np.sum(data_copy['mask'], axis=0)
    else:
        weights = None

    # data_copy['data'] = np.average(data_copy['data'], weights=weights,
    #                                axis=0)
    data_copy['data'] = my_average(data_copy['data'], weights, axis=0)

    if 'errors' in data_copy:
        data_copy['errors'] = np.sqrt(my_average(data_copy['errors']**2,
                                                 weights=weights, axis=0))
    return data_copy


def prerpare_header(data):
    if 'headers' not in data:
        header = fits.header.Header()
    else:
        header = data['headers'][0]
    if 'keys' in data:
        header.update(data['keys'])
    return header


def pipeline(frames, headers=None, bias_obj=None, flat_obj=None, dark_obj=None,
             ch_obj=None, xcorr_obj=None, ycorr_obj=None, wl_obj=None,
             sky_obj=None):

    data = {'data': frames, 'headers': headers, 'errors': np.sqrt(frames),
            'mask': np.zeros_like(frames).astype(bool), 'keys': dict()}
    # data = {'data': frames, 'headers': headers}
    print("Processing...")
    data_no_bias = bias.process_bias(data, bias_obj)
    print("BIAS")
    data_no_dark = dark.process_dark(data_no_bias, dark_obj)
    print("DARK")
    data_no_flat = flat.process_flat(data_no_dark, flat_obj)
    print("FLAT")
    data_no_cosmics = cosmics.process_cosmics(data_no_flat, ch_obj, bias_obj)
    print("COSMICS")
    data_wl_corrected = dispersion.process_dispersion(data_no_cosmics, wl_obj)
    print("DISPERSION")
    if ycorr_obj != 'obj':
        data_dist_corrected = distorsion.process_distorsion(data_wl_corrected,
                                                            ycorr_obj)
    else:
        dist_file = distorsion.get_distorsion_file(data['data'], bias_obj, dark_obj,
                                    flat_obj, ch_obj, wl_obj)
        ycorr_obj = distorsion.distorsion_from_file(dist_file)
        data_dist_corrected = distorsion.process_distorsion(data_wl_corrected,
                                                            ycorr_obj)
    print("DISTORSION")
    data_sum = sum_data(data_dist_corrected)
    print("SUM")

    # hdr = prerpare_header(data_sum)

    # result = [fits.PrimaryHDU(data_sum['data'], header=hdr)]
    # if 'errors' in data_sum:
    #     result.append(fits.ImageHDU(data_sum['errors'], name='errors'))
    # if 'mask' in data_sum:
    #     result.append(fits.ImageHDU(data_sum['mask'], name='mask'))

    result = fits_from_data(data_sum)

    return result


def read_yaml_obj(yaml_name):
    file = open(yaml_name, 'r')
    config = yaml.load(file, Loader=yaml.SafeLoader)
    file.close()

    if 'object' not in config:
        return None

    bias_obj = dark_obj = flat_obj = ch_obj = xcorr_obj = ycorr_obj = \
        wl_obj = sky_obj = file_names = outname = None

    if 'output' in config['object']:
        outname = config['object']['output']

    file_names = config['object']['filenames']

    if 'additional' in config['object']:
        adds = config['object']['additional']
    else:
        adds = []

    if 'B' in adds:
        bias_obj = bias.bias_from_file(adds['B'])
    if 'D' in adds:
        dark_obj = dark.dark_from_file(adds['D'])
    if 'F' in adds:
        flat_obj = flat.flat_from_file(adds['F'])
    if 'C' in adds:
        ch_obj = adds['C']
    if 'X' in adds:
        xcorr_obj = corrections.corrections_from_file(adds['X'])
    if 'Y' in adds:
        if adds['Y']:
            ycorr_obj = distorsion.distorsion_from_file(adds['Y'])
        else:
            ycorr_obj = 'obj'
    if 'W' in adds:
        wl_obj = dispersion.dispersion_from_file(adds['W'])
    # if 'sky' in config:
    #     sky_obj = config['sky']
    #     if 'skyflat' in sky_obj:
    #         sky_obj['skyflat'] = cal + sky_obj['skyflat']

    return(file_names, bias_obj, dark_obj, flat_obj, ch_obj, xcorr_obj,
           ycorr_obj, wl_obj, sky_obj, outname)


def prepare_configs(yaml_name):
    file = open(yaml_name, 'r')
    config = yaml.load(file, Loader=yaml.SafeLoader)
    file.close()

    seq_calibs = ['bias', 'dark', 'corr', 'flat', 'disp', 'dist']
    process = dict()

    for calib in seq_calibs:
        if calib in config:
            if 'rawfiles' in config[calib]:
                argstring = [calib]
                argstring.extend(config[calib]['rawfiles'])
                for k in config[calib]['additional'].keys():
                    argstring.append('-' + k)
                    if k != 'C':
                        argstring.append(config[calib]['additional'][k])
                argstring.append('-' + 'o')
                argstring.append(config[calib]['calibration'])
                process[calib] = argstring

    if 'bias' in process:
        bias.main(process['bias'])
        print("BIAS CALIB")
    if 'dark' in process:
        dark.main(process['dark'])
        print("DARK CALIB")
    if 'corr' in process:
        corrections.main(process['corr'])
        print("GEOMETRY CALIB")
    if 'flat' in process:
        flat.main(process['flat'])
        print("FLAT CALIB")
    if 'disp' in process:
        dispersion.main(process['disp'])
        print("DISPERSION CALIB")
    if 'dist' in process:
        if config['dist']['calibration']:
            distorsion.main(process['dist'])
        print("DISTORSION CALIB")
    return ('object' in config)





def main(args=None):
    parser = argparse.ArgumentParser()
    default_out = '../data/result.fits'
    parser.add_argument('filenames', nargs='+',
                        help="""fits files with spectra to reduce OR yaml file
                        with configuration. Note: optional arguments with
                        calibrations will overwrite yaml config""")
    parser.add_argument('-d', '--dir', help="directory with input files",
                        default='')
    parser.add_argument('-c', '--cal', help="directory with calibration files",
                        default='')
    parser.add_argument('-y', '--yaml', help="yaml file to save configuration")
    parser.add_argument('-o', '--out', default=default_out,
                        help='output file')
    parser.add_argument('-B', '--BIAS', help="bias frame (fits) to substract")
    parser.add_argument('-D', '--DARK',
                        help="prepared fits-file with dark frames")
    parser.add_argument('-F', '--FLAT',
                        help="prepared fits-file with flat frame")
    parser.add_argument('-C', '--COSMICS', action='store_true',
                        help="set this argument to clear cosmic hints")
    parser.add_argument('-X', '--GEOMETRY',
                        help="file with correction map (x-axis)")
    parser.add_argument('-Y', '--DISTORSION',
                        help="file with correction map (y-axis)")
    parser.add_argument('-W', '--WAVELENGTH',
                        help="file with correction map (wavelength)")
    parser.add_argument('-S', '--SKY', nargs='*',
                        help="""[file] Y1 Y2 [Y3 Y4] - file with skyflat
                        (optional) and regions to subtract sky""")

    pargs = parser.parse_args(args[1:])

    bias_obj = dark_obj = flat_obj = ch_obj = xcorr_obj = ycorr_obj = \
        wl_obj = sky_obj = outname = None

    first_arg = pargs.filenames[0]
    farg_ext = first_arg.split('.')[-1]
    if farg_ext == 'yaml' or farg_ext == 'yml':
        if_obj = prepare_configs(first_arg)
        if if_obj:
            file_names, bias_obj, dark_obj, flat_obj, ch_obj, xcorr_obj, \
                ycorr_obj, wl_obj, sky_obj, outname = read_yaml_obj(first_arg)
        else:
            return(0)
    else:
        file_names = pargs.filenames
        file_names = [pargs.dir + fn for fn in file_names]

    if outname is None:
        outname = pargs.out
    elif pargs.out != default_out:
        outname = pargs.out

    cal = pargs.cal

    if pargs.BIAS:
        bias_obj = bias.bias_from_file(cal + pargs.BIAS)
    if pargs.DARK:
        dark_obj = dark.dark_from_file(cal + pargs.DARK)
    if pargs.FLAT:
        flat_obj = flat.flat_from_file(cal + pargs.FLAT)
    if pargs.COSMICS:
        ch_obj = True
    if pargs.GEOMETRY:
        xcorr_obj = corrections.corrections_from_file(cal + pargs.GEOMETRY)
    if pargs.DISTORSION:
        ycorr_obj = distorsion.distorsion_from_file(cal + pargs.DISTORSION)
    if pargs.WAVELENGTH:
        wl_obj = dispersion.dispersion_from_file(cal + pargs.WAVELENGTH)
    if pargs.SKY:
        try:
            skyborders = [int(x) for x in pargs.SKY]
            sky_obj = {'skyflat': None, 'borders': skyborders}
        except ValueError:
            skyfile = sky.sky_from_file(pargs.SKY[0])
            skyborders = [int(x) for x in pargs.SKY[1:]]
            sky_obj = {'skyflat': skyfile, 'borders': skyborders}

    data, headers = open_fits_array_data(file_names, header=True)

    result = pipeline(data, headers, bias_obj, flat_obj, dark_obj, ch_obj,
                      xcorr_obj, ycorr_obj, wl_obj, sky_obj)
    print('writeto ', outname)
    result.writeto(outname, overwrite=True)
    return(0)


if __name__ == '__main__':
    import sys
    from genfuncs import open_fits_array_data
    import argparse
    sys.exit(main(sys.argv))
