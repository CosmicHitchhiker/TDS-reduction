import numpy as np
import scipy.optimize as opt
import astropy.signal as sig
from itertools import zip_longest, islice, cycle
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from tqdm import tqdm


def gauss(x, s, x0, A):
    """Simple gaussian"""
    return A * np.exp(-(x - x0)**2 / (2 * s**2))


def gauss_cont(x, s, x0, A, k, b):
    """Gaussian plus continuum"""
    return A * np.exp(-(x - x0)**2 / (2 * s**2)) + k * x + b


# FIXME: make rng wider, fit by gauss_cont
def one_peak_fwhm(x, A, wl, spec, guess=1):
    rng = (wl > x - guess) & (wl < x + guess)
    return 2.355 * np.abs(opt.curve_fit(gauss, wl[rng], spec[rng],
                                        p0=[guess, x, A])[0][0])


# FIXME: fit by gauss_cont
def one_peak_amp(x, A, wl, spec, fwhm=10):
    rng = (wl > x - 2 * fwhm) & (wl < x + 2 * fwhm)
    return 2.355 * np.abs(opt.curve_fit(gauss, wl[rng], spec[rng],
                                        p0=[fwhm, x, A])[0][2])


def calc_fwhm(spec, wl=None, n=3, guess=10):
    if wl is None:
        wl = np.arange(len(spec))
    peaks = sig.find_peaks(spec)[0]
    amps = spec[peaks]
    peaks = peaks[np.argsort(amps)][-n:]
    amps = amps[np.argsort(amps)][-n:]

    def one_peak_fwhm_(x, A):
        return one_peak_fwhm(x, A, wl, spec, guess)

    fwhm = np.average(list(map(one_peak_fwhm_, wl[peaks], amps)))
    return fwhm


def find_peaks(spec, fwhm=1, h=1, d=1):
    '''Ищет пики выше заданного уровня h относительно медианы.
    Затем удаляет из списка пики, у которых есть соседи ближе fwhm*d'''
    # spec = spec-np.min(spec)
    spec = spec / np.median(spec)
    pks = sig.find_peaks(spec, height=h, distance=fwhm * d)[0]
    return(pks[:])


def find_lines_cluster(peaks, y=None, verbose=False, k=50, eps=70, clust=10):
    "peaks - координаты пиков в каждой строчке"
    if y is None:
        y = np.arange(len(peaks))

    # После zip индексация 0 индекс - номер "линии"
    # В массиве - х-координаты пиков
    peaks = np.array(list(zip_longest(*peaks)), dtype='float')
    # у-координата для каждой х-координаты пика
    y_matrix = np.tile(y, peaks.shape[0])
    # Плоский массив х-координат
    # k показывает "вес" сдвига по х
    peaks_f = peaks.flatten() * k
    # Убираем все NaN
    mask = np.isnan(peaks_f)
    peaks_f = peaks_f[~mask]
    y_matrix = y_matrix[~mask]
    # Массив координат (х,у) всех найденых пиков
    vectors = np.array([peaks_f, y_matrix]).T

    clustering = DBSCAN(eps=eps, min_samples=clust).fit(vectors)
    y_pred = clustering.labels_.astype('int')

    vectors[:, 0] /= k

    if verbose:
        plt.figure()
        plt.clf()
        plt.title('peaks clusters')
        clrs = "377eb8 ff7f00 4daf4a f781bf a65628 984ea3 999999 e41a1c dede00"
        clrs = ["#" + c for c in clrs.split()]
        colors = np.array(list(islice(
            cycle([*clrs, ]), int(max(y_pred) + 1),)))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(vectors[:, 0], vectors[:, 1], s=10, color=colors[y_pred])
        plt.show()

    mask = (y_pred >= 0)  # убираем не классифицированные точки

    # убираем короткие линии
    for n_line in set(y_pred):
        line_y = vectors[:, 1][y_pred == n_line]
        dy = line_y.max() - line_y.min()
        if (dy < (y.max() - y.min()) / 2.5):
            mask = (mask & (y_pred != n_line))

    return(vectors[mask], y_pred[mask])


def fine_peak_position_i(row, peak, fwhm=10, x=None):
    if x is None:
        x = np.arange(len(row), dtype='float')
    peak_f = peak
    peak = int(peak)
    fwhm = int(fwhm)

    amp = row[peak]
    b = np.median(row) / 2.
    x = x[peak - 2 * fwhm:peak + 2 * fwhm]
    y = row[peak - 2 * fwhm:peak + 2 * fwhm]
    try:
        p0 = [fwhm / 2.335, peak, amp, 0, b]
        bounds = ([fwhm * 0.3 / 2.335, peak - 1, amp * 0.7, -0.1, 0],
                  [fwhm * 3 / 2.335, peak + 1, amp * 1.3, 0.1, np.inf])
        fine_peak = opt.curve_fit(gauss_cont, x, y, p0=p0,
                                  bounds=bounds)[0][1]
        return(fine_peak)
    except RuntimeError:
        return(peak_f)


def refine_peaks_i(neon, peaks, fwhm=10):
    # peaks[i] - это (x_i, y_i)
    x = np.arange(len(neon[0]))
    # Для каждого из пиков уточняем его позицию
    for i in tqdm(range(len(peaks))):
        peak = peaks[i]
        peaks[i][0] = fine_peak_position_i(
            neon[int(peak[1])], peak[0], fwhm, x)
    return(peaks)


def my_poly(p, y):
    '''Applying polinomial to an array of values.

    //MORE DETAILED DESCRIPTION IS COMING///

    Parameters
    ----------
    p : ndarray
        Vector or matrix of polinomial coefficients
    y : float or ndarray
        Value or an array of values to which polinomial
        will be applied.

    Returns
    -------
    k : float or array of floats
        Result:
        p - vector, y - float -> float
        p - matrix, y - float -> vector
        p - vector, y - vector -> vector
        p - matrix, y - vector -> matrix
    '''
    n = len(p)
    m = len(y)
    pow_arr = np.arange(n - 1, -1, -1)
    y = np.ones((n, m)) * y
    y_powered = np.power(y.T, pow_arr)
    return np.dot(y_powered, p)
