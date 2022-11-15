#! /usr/bin/python3

import numpy as np
import geometry as gm
from matplotlib import pyplot as plt
import bias
import dark
import cosmics
from astropy.io import fits
import argparse
from genfuncs import open_fits_array_data


def get_peaks_clust(neon, h=10, d=5, k=50, eps=70, clust=10):
    h = h  # Во сколько минимально раз пик должен быть выше медианы
    d = d  # Минимальное расстояние (в fwhm) между пиками

    y, x = np.shape(neon)
    y = np.arange(y)
    x = np.arange(x)

    # За fwhm считаем fwhm (в пикселях) средней (по Y) строки
    fwhm = gm.calc_fwhm(neon[int(len(neon) / 2)])
    print('fwhm = ', fwhm, 'pix')

    # Пики в каждой строчке (list из ndarray разной длины)
    peaks = [gm.find_peaks(row, fwhm=fwhm, h=h, d=d) for row in neon]

    peaks, n_lines = gm.find_lines_cluster(peaks, y, verbose=True, k=k,
                                           eps=eps, clust=clust)
    return peaks, n_lines, fwhm


def get_peaks_clust_setup(neon):

    h = 10
    d = 5
    k = 50
    eps = 70
    clust = 10

    need_to_change = 'Yes'
    while need_to_change != '':
        peaks, n_lines, fwhm = get_peaks_clust(neon, h, d, k, eps, clust)

        params = argparse.ArgumentParser(exit_on_error=False)
        params.add_argument('-l', type=float, default=h)
        params.add_argument('-d', type=float, default=d)
        params.add_argument('-k', type=float, default=k)
        params.add_argument('-eps', type=float, default=eps)
        params.add_argument('-clust', type=int, default=clust)
        parags = params.parse_args('')
        print(parags)
        need_to_change = input("Change any parameters?(leave blank if No)")
        if need_to_change:
            parags = params.parse_args(need_to_change.split())
            h = parags.l
            d = parags.d
            k = parags.k
            eps = parags.eps
            clust = parags.clust
    return peaks, n_lines, fwhm


def get_correction_map(neon, verbose=False, ref='mean', use_clust=True, h=10,
                       d=20, return_peaks=False):
    '''Считает карту интерполяции.
    В каждой строчке - те координаты, на которые нужно
    интерполировать исходное изображение, чтобы исправить
    геометрические искажения вдоль оси Х.
    (Опорные кадры - линейчатые спектры газов)
    ref = 'mean' - приводить к средним значениям
    ref = 'center' - приводить к значению в центре кадра
    '''
    # h = 10  # Во сколько минимально раз пик должен быть выше медианы
    # d = 20  # Минимальное расстояние (в fwhm) между пиками

    y, x = np.shape(neon)
    y = np.arange(y)
    x = np.arange(x)

    peaks, n_lines, fwhm = get_peaks_clust_setup(neon)

    # За fwhm считаем fwhm (в пикселях) средней (по Y) строки
    # fwhm = gm.calc_fwhm(neon[int(len(neon) / 2)])
    # print(('fwhm = ', fwhm, 'pix') if verbose else '')

    # # Пики в каждой строчке (list из ndarray разной длины)
    # def find_peaks_(row):
    #     return(gm.find_peaks(row, fwhm=fwhm, h=h, d=d))
    # peaks = list(map(find_peaks_, neon))
    # print('***all peaks are found***' if verbose else '')
    # peaks, n_lines = gm.find_lines_cluster(peaks, y, verbose=True)
    # В каждом элементе peaks 1-я координата - х, 2 - у
    # в n_lines для каждой точки записано к какой она линии относится
    # peaks = gm.refine_peaks_i(neon, peaks, fwhm)

    # Нумеруем каждую найденную линию неона (делаем список "номеров")
    enum_lines = set(n_lines.tolist())
    # print(peaks[n_lines==list(enum_lines)[0]])

    # Полиномом какой степени фитируется каждая искривлённая линия
    # Полиномом какой степени фитируется каждый полином (deg2)
    # (в зависимости от х-координаты центра линии)
    if len(enum_lines) > 3:
        deg = 2
        deg2 = 3
    else:
        deg = 3
        deg2 = 1

    k = np.zeros((len(enum_lines), deg + 1))
    plt.figure(18)
    plt.clf()
    plt.imshow(neon)
    for i, n in enumerate(enum_lines):
        line = peaks[n_lines == n].T
        plt.plot(line[0], line[1], '.')
        k[i] = np.polyfit(line[1], line[0], deg)
        plt.plot(np.polyval(k[i], y), y)
        # print(k[i])
    plt.show()

    med_y = np.median(y)  # номер (у-координата)средней строчки
    # Для каждой из линий её предсказанная х-координата в средней строчке
    mean_peaks = np.array(list(map(lambda x: np.polyval(x, med_y), k)))

    if len(enum_lines) > 1:
        corr = np.polyfit(mean_peaks, k, deg2)
        corr_map = gm.my_poly(gm.my_poly(corr, x).T, y)
    else:
        corr = np.polyval(k[0], y)
        yg, xg = np.mgrid[:len(neon), :len(neon[0])]
        print(xg)
        xg = xg - mean_peaks[0]
        print(xg)
        corr_map = (xg.T + corr).T
        print(corr)
        print(corr_map)

    plt.imshow(corr_map)
    plt.show()
    good_columns = (np.min(corr_map, axis=0) > 0)
    # Умножение для bool - это and!
    good_columns *= (np.max(corr_map, axis=0) < x[-1])

    new_x = x[good_columns].astype('int')
    if return_peaks:
        return corr_map, new_x, mean_peaks

    return(corr_map, new_x)


def interpolate_correction_map(frame, corr_map, inverse=False):
    x = np.arange(len(frame[0]))
    if inverse:
        data = np.array(list(map(lambda val, coord: np.interp(x, coord, val),
                                 frame, corr_map)))
    else:
        data = np.array(list(map(lambda val, coord: np.interp(coord, x, val),
                                 frame, corr_map)))
    return data


def get_corrections_file(data, bias_obj=None, dark_obj=None, cosm_obj=None):
    data_copy = {'data': data.copy()}
    data_copy = bias.process_bias(data_copy, bias_obj)
    if cosm_obj:
        data_copy = cosmics.process_cosmics(data_copy)

    neon = np.sum(data_copy['data'], axis=0)
    corr_map, new_x = get_correction_map(neon)
    res = fits.PrimaryHDU(corr_map)
    res.header['GOODX1'] = np.min(new_x)
    res.header['GOODX2'] = np.max(new_x)

    corr_obj = corrections_from_file(res)
    data = {'data': [neon]}
    res2 = process_corrections(data, corr_obj)['data']
    res2 = fits.ImageHDU(res2[0])
    res2.name = 'neon'
    return fits.HDUList([res, res2])


def corrections_from_file(corrections_file):
    if isinstance(corrections_file, str):
        corrections_file = fits.open(corrections_file)[0]
    elif isinstance(corrections_file, fits.HDUList):
        corrections_file = corrections_file[0]

    x1 = corrections_file.header['GOODX1']
    x2 = corrections_file.header['GOODX2'] + 1
    good_x = np.arange(x1, x2)
    res = {'data': corrections_file.data, 'new_x': good_x}
    return res


def process_corrections(data, corr_obj):
    data_copy = data.copy()
    if corr_obj is None:
        return data_copy
    new_x = corr_obj['new_x']
    corr_map = corr_obj['data'][:, new_x]
    data_copy['data'] = np.array([interpolate_correction_map(x, corr_map)
                                  for x in data_copy['data']])
    return data_copy


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+',
                        help="fits files with neon frames")
    parser.add_argument('-d', '--dir', help="directory with input files")
    parser.add_argument('-o', '--out', default='../data/correction_map.fits',
                        help='output file')
    parser.add_argument('-B', '--BIAS', help="bias frame (fits) to substract")
    parser.add_argument('-D', '--DARK',
                        help="prepared fits-file with dark frames")
    parser.add_argument('-C', '--COSMICS', action='store_true',
                        help="set this argument to clear cosmic hints")
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

    arc_names = pargs.filenames
    if pargs.dir:
        arc_names = [pargs.dir + x for x in arc_names]
    arc_files, headers = open_fits_array_data(arc_names, header=True)

    corr_file = get_corrections_file(arc_files, bias_obj, dark_obj,
                                     if_clear_cosmics)
    corr_obj = corrections_from_file(corr_file)
    data = {'data': arc_files}
    res = process_corrections(data, corr_obj)
    plt.imshow(res['data'][0])
    plt.show()
    corr_file.writeto(pargs.out, overwrite=True)
    return(0)


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
