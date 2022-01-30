import numpy as np
from astropy.io import fits


def open_fits_array_data(names, margins=[0, None, 0, None], header=False):
    '''Give an array of data of named fits files.

    Read data from all mentioned files and return an 3D ndarray of images.
    If margins are mentioned, evry image is cropped.
    If it is necessary also return a list of headers.

    Parameters
    ----------
    names : list of strings
        List of files to be opened.
    margins : list of inegers, optional
        Coordinates of image rectangle to be returned: xmin, xmax, ymin, ymax.
    header : bool, optional
        If True, also return list of headers.

    Returns
    -------
    fitses : 3D ndarray
        An array of opened fits-images.
    headers : list of fits.header
        A list of heders of mentioned files (returned when header is True).
    '''
    xmin, xmax, ymin, ymax = margins
    # Some fits files store image as 2D array and other - as 3D array.
    # So we dermine number of dimentions and if it's equal to 3 only first
    # 2D array will be used
    if len(np.shape(fits.getdata(names[0]))) == 2:
        fitses = np.array([fits.getdata(names[0])[ymin:ymax, xmin:xmax]])
        if header is True:
            headers = [fits.getheader(names[0])]
        for name in names[1:]:
            cutted_data = [fits.getdata(name)[ymin:ymax, xmin:xmax]]
            fitses = np.append(fitses, cutted_data, axis=0)
            if header is True:
                headers.append(fits.getheader(name))
    elif len(np.shape(fits.getdata(names[0]))) == 3:
        fitses = np.array([fits.getdata(names[0])[0, ymin:ymax, xmin:xmax]])
        if header is True:
            headers = [fits.getheader(names[0])]
        for name in names[1:]:
            cutted_data = [fits.getdata(name)[0, ymin:ymax, xmin:xmax]]
            fitses = np.append(fitses, cutted_data, axis=0)
            if header is True:
                headers.append(fits.getheader(name))
    if header is True:
        return fitses, headers
    else:
        return fitses
