#! /usr/bin/python3


import numpy as np
import geometry as gm
from matplotlib import pyplot as plt


def get_correction_map(neon, verbose=False, ref='mean', use_clust=True):
    '''Считает карту интерполяции.
    В каждой строчке - те координаты, на которые нужно
    интерполировать исходное изображение, чтобы исправить
    геометрические искажения вдоль оси Х.
    (Опорные кадры - линейчатые спектры газов)
    ref = 'mean' - приводить к средним значениям
    ref = 'center' - приводить к значению в центре кадра
    '''
    h = 10  # Во сколько минимально раз пик должен быть выше медианы
    d = 20  # Минимальное расстояние (в fwhm) между пиками

    y, x = np.shape(neon)
    y = np.arange(y)
    x = np.arange(x)

    # За fwhm считаем fwhm (в пикселях) средней (по Y) строки
    fwhm = gm.calc_fwhm(neon[int(len(neon) / 2)])
    print(('fwhm = ', fwhm, 'pix\n') if verbose else '', end='')

    # Пики в каждой строчке (list из ndarray разной длины)
    def find_peaks_(row):
        return(gm.find_peaks(row, fwhm=fwhm, h=h, d=d))
    peaks = list(map(find_peaks_, neon))
    print('***all peaks are found***' if verbose else '')
    # if verbose:
    #     plt.figure(10)
    #     plt.clf()
    #     plt.title("raw find_peaks")
    #     plt.imshow(neon)
    #     for i in y:
    #         plt.plot(peaks[i], np.ones(len(peaks[i]))*y[i], '.')
    #     plt.show()
    peaks, n_lines = gm.find_lines_cluster(peaks, y, verbose=True)
    # В каждом элементе peaks 1-я координата - х, 2 - у
    # в n_lines для каждой точки записано к какой она линии относится
    peaks = gm.refine_peaks(neon, peaks, fwhm)

    # Нумеруем каждую найденную линию неона (делаем список "номеров")
    enum_lines = set(n_lines.tolist())
    # print(peaks[n_lines==list(enum_lines)[0]])

    # Полиномом какой степени фитируется каждая искривлённая линия
    deg = 2
    # Полиномом какой степени фитируется каждый полином
    # (в зависимости от х-координаты центра линии)
    deg2 = 3

    k = np.zeros((len(enum_lines), deg+1))
    plt.figure(18)
    plt.clf()
    plt.imshow(neon)
    for i, n in enumerate(enum_lines):
        line = peaks[n_lines == n].T
        plt.plot(line[0], line[1], '.')
        k[i] = np.polyfit(line[1], line[0], deg)
        plt.plot(np.polyval(k[i], y), y)
        print(k[i])
    plt.show()

    med_y = np.median(y)  # номер (у-координата)средней строчки
    # Для каждой из линий её предсказанная х-координата в средней строчке
    mean_peaks = np.array(list(map(lambda x: np.polyval(x, med_y), k)))

    corr = np.polyfit(mean_peaks, k, deg2)
    corr_map = gm.my_poly(gm.my_poly(corr, x).T, y)

    good_columns = (np.min(corr_map, axis=0) > 0)
    # Умножение для bool - это and!
    good_columns *= (np.max(corr_map, axis=0) < x[-1])

    new_x = x[good_columns].astype('int')

    return(corr_map, new_x)


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
    pargs = parser.parse_args(args[1:])

    if pargs.BIAS:
        superbias = fits.getdata(pargs.BIAS)
    else:
        superbias = 0

    # if pargs.DARK:
    #
    return(0)


if __name__ == '__main__':
    import sys
    from utils import open_fits_array_data
    from astropy.io import fits
    import argparse
    sys.exit(main(sys.argv))
