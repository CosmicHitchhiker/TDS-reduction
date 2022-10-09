#! /usr/bin/python3

import numpy as np


def extract_sky(data, sky):
    '''Remove sky spectrum from the image.

    Read sky spectrum from the mentioned area.
    Fit sky spectrum changing by the second-order polinomial.
    Substract sky from the object area.

    Parameters
    ----------
    data : 2D ndarray
        Fits image
    sky : array of  2*n integers
        Numbers of strings to be used as borders of sky area

    Returns
    -------
    data : 2D ndarray
        Image of object with sky spectrum substracted
    '''
    y_sky = np.arange(sky[0], sky[1])
    for i in range(2, len(sky), 2):
        if sky[i] > len(data):
            break
        if sky[i + 1] > len(data):
            sky[i + 1] = len(data) - 1
        y_sky = np.append(y_sky, np.arange(sky[i], sky[i + 1]))
    tdata = data[y_sky].T
    sky_poly = np.array(list(map(lambda x: np.polyfit(y_sky, x, 2), tdata)))
    real_sky = np.array(list(map(lambda x: np.polyval(x, np.arange(len(data))), sky_poly))).T
    return data - real_sky, real_sky


def process_sky(data, sky_y):
    data_copy = data.copy()
    res = []
    res_sky = []
    for frame in data_copy['data']:
        r, r_sky = extract_sky(frame, sky_y)
        res.append(r)
        res_sky.append(r_sky)
    data_copy['data'] = res
    data_copy['sky'] = res_sky
    return(data_copy)


def main(args=None):
    """This method runs if the file is running as a program"""
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+',
                        help="files to clear cosmic rays in them")
    parser.add_argument('-d', '--dir', help="directory with input files")
    parser.add_argument('-o', '--out', default='../data/',
                        help='directory to save results')
    parser.add_argument('-B', '--BIAS', help="bias frame (fits) to subtract")
    pargs = parser.parse_args(args[1:])
    print(pargs)

    # if pargs.BIAS:
    #     bias_obj = bias.bias_from_file(pargs.BIAS)
    # else:
    #     bias_obj = None

    # # print(args)
    # frame_names = pargs.filenames
    # if pargs.dir:
    #     frame_names = [pargs.dir + x for x in frame_names]

    # frame_files, headers = open_fits_array_data(frame_names, header=True)
    # processed_files = get_cosmics_file(frame_files, headers, bias_obj)
    # for name, hdu in zip(frame_names, processed_files):
    #     name_to_write = (name.split('/')[-1]).split('.')[0]
    #     name_to_write = pargs.out + name_to_write + '_nocosmics.fits'
    #     hdu.writeto(name_to_write, overwrite=True)
    # superbias_file = get_bias_file(bias_files, headers[0])
    # superbias_file.writeto(pargs.out, overwrite=True)
    return 0


if __name__ == '__main__':
    import sys
    # from utils import open_fits_array_data
    import argparse
    sys.exit(main(sys.argv))
