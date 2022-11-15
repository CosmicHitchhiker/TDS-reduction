#! /usr/bin/python3
"This module contains functions to detect cosmic rays"

from lacosmic import lacosmic
import numpy as np
from astropy.io import fits
import bias


def process_cosmics(data, ch_obj=True, bias_obj=None):
    """Clear csomic hints at given images

    Apply lacosmic method to every frame in given data.

    Parameters
    ----------
    data : dict
        'data' - 3D ndarray, array of data images

    Returns
    -------
    data_copoy : dict
        Has the same structure as input data
    """
    data_copy = data.copy()
    if not ch_obj:
        return data_copy

    frames = data_copy['data']

    if 'mask' in data_copy:
        msk = data_copy['mask']
    else:
        msk = [None] * len(data_copy['data'])

    if 'errors' in data_copy:
        err = data_copy['errors']
    else:
        err = [None] * len(data_copy['data'])

    if bias_obj is not None:
        bias = bias_obj['data']
    else:
        bias = None

    readnois = 3.
    if 'keys' in data_copy:
        if 'READNOIS' in data_copy['keys']:
            readnois = data_copy['keys']['READNOIS']

    def prepared_lacosmic(frame, errors, mask):
        # norm = simple_norm(frame, 'log', percent=95)
        # plt.imshow(frame, origin='lower', norm=norm)
        # plt.show()
        return lacosmic(frame, 1.7, 7, 3, effective_gain=1, readnoise=readnois,
                        error=errors, mask=mask, background=bias)

    resframes = [prepared_lacosmic(x, y, z) for x, y, z in
                 zip(frames, err, msk)]
    data_copy['data'] = np.array([x[0] for x in resframes])
    if 'mask' in data_copy:
        for i in range(len((data_copy['mask']))):
            data_copy['mask'][i] = data_copy['mask'][i] | resframes[i][1]
    return data_copy


def get_cosmics_file(frames, headers, bias_obj):
    """Remove cosmic images from given frames

    Subtract bias frame from every image with data.
    Apply lacosmic algorythm to them.
    Fill headers with comment about this procedure.

    Parameters
    ----------
    frames : 3D ndarray
        frames[i] - frame with image data
    headers : astropy header
        corresponding headers for each frame
    bias_obj : dict
        object to pass to bias.process_bias

    Returns
    -------
    res : list of fits.PrimaryHDU
        Resulting files with cosmic hints removed
    """
    data = {'data': frames}
    data_no_bias = bias.process_bias(data, bias_obj)
    data_no_cosmics = process_cosmics(data_no_bias)
    res = []
    for frame, hdr in zip(data_no_cosmics['data'], headers):
        hdr['COMMENT'] = 'Cosmics removed with lacosmic algorythm'
        res.append(fits.PrimaryHDU(frame, header=hdr))
    return res


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

    if pargs.BIAS:
        bias_obj = bias.bias_from_file(pargs.BIAS)
    else:
        bias_obj = None

    # print(args)
    frame_names = pargs.filenames
    if pargs.dir:
        frame_names = [pargs.dir + x for x in frame_names]

    frame_files, headers = open_fits_array_data(frame_names, header=True)
    processed_files = get_cosmics_file(frame_files, headers, bias_obj)
    for name, hdu in zip(frame_names, processed_files):
        name_to_write = (name.split('/')[-1]).split('.')[0]
        name_to_write = pargs.out + name_to_write + '_nocosmics.fits'
        hdu.writeto(name_to_write, overwrite=True)
    # superbias_file = get_bias_file(bias_files, headers[0])
    # superbias_file.writeto(pargs.out, overwrite=True)
    return 0


if __name__ == '__main__':
    import sys
    from genfuncs import open_fits_array_data
    import argparse
    sys.exit(main(sys.argv))
