#! /usr/bin/python3

import numpy as np
from genfuncs import open_fits_array_data
from astropy.io import fits
from astropy.visualization import simple_norm
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton


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
    data_copy = data.copy()
    y_sky = np.arange(sky[0], sky[1])
    for i in range(2, len(sky), 2):
        if sky[i] > len(data_copy):
            break
        if sky[i + 1] > len(data_copy):
            sky[i + 1] = len(data_copy) - 1
        y_sky = np.append(y_sky, np.arange(sky[i], sky[i + 1]))
    tdata = data_copy[y_sky].T
    sky_poly = np.array(list(map(lambda x: np.polyfit(y_sky, x, 2), tdata)))
    real_sky = [np.polyval(x, np.arange(len(data_copy))) for x in sky_poly]
    real_sky = np.array(real_sky).T
    data_nosky = data - real_sky
    return data_nosky, real_sky


def process_sky(data, sky_y):
    data_copy = data.copy()
    res = []
    res_sky = []
    for i, frame in enumerate(data_copy['data']):
        r, r_sky = extract_sky(frame, sky_y)
        res.append(r)
        res_sky.append(r_sky)
        if 'mask' in data_copy:
            data_copy['mask'][i] = data_copy['mask'][i] | (r < 0)
    data_copy['data'] = res
    data_copy['sky'] = res_sky
    return(data_copy)


def get_sky_borders(frame):

    fig, ax = plt.subplots(1, 1)
    norm_im = simple_norm(frame, 'linear', percent=95)
    ax.imshow(frame, cmap='gray', norm=norm_im, origin='lower')
    plt.ion()
    result = []
    lines = []

    def onclick(event):
        if event.button is MouseButton.LEFT:
            py = int(event.ydata)
            lines.append(ax.axhline(y=py, color='b', linestyle='-'))
            result.append(py)
        if (event.button is MouseButton.RIGHT) and len(result) > 0:
            py = event.ydata
            npdiff = np.abs(np.array(result) - py)
            if np.min(npdiff) < 15:
                ind = np.argmin(npdiff)
                result.pop(ind)
                try:
                    ax.lines[ind].remove()
                except IndexError:
                    pass
                lines.pop(ind)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)
    fig.canvas.mpl_disconnect(cid)
    plt.ioff()

    result = np.sort(result).tolist()
    if len(result) % 2 != 0:
        raise ValueError("Number of sky borders should be even!")
    print("Sky borders: ", result)
    return result



def main(args=None):
    """This method runs if the file is running as a program"""
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help="file to clear sky in it")
    parser.add_argument('-y', '--ysky', nargs='+',
                        help='y coordinates (px) to sample sky')
    parser.add_argument('-o', '--out',
                        help='result file')
    parser.add_argument('-s', '--skyfile',
                        help='filename to save extracted sky')
    pargs = parser.parse_args(args[1:])

    if pargs.out:
        resname = pargs.out
    else:
        resname = pargs.filename.split('.')[-2] + '_nosky.fits'

    

    frame = fits.open(pargs.filename)

    if pargs.ysky:
        sky_y = [int(y) for y in pargs.ysky]
    else:
        sky_y = get_sky_borders(frame[0].data)

    data = {'data': [frame[0].data]}
    if 'mask' in frame:
        data['mask'] = [(frame['mask'].data == 1)]
    data_copy = process_sky(data, sky_y)

    frame[0].data = data_copy['data'][0]
    if 'mask' in frame:
        frame['mask'].data = data_copy['mask'][0].astype(int)
    frame.writeto(resname, overwrite=True)

    if pargs.skyfile:
        skyframe = fits.PrimaryHDU(data_copy['sky'])
        keys_to_copy = ['CRPIX1', 'CRVAL1', 'CDELT1', 'CTYPE1', 'CRDER1',
                        'FWHM', 'CRPIX2A', 'CRPIX2B', 'CRPIX2', 'CRVAL2',
                        'CDELT2']
        for key in keys_to_copy:
            skyframe.header[key] = frame[0].header[key]
        skyframe.writeto(pargs.skyfile, overwrite=True)

    return 0


open_fits_array_data
if __name__ == '__main__':
    import sys
    # from utils import open_fits_array_data
    import argparse
    sys.exit(main(sys.argv))
