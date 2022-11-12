import numpy as np
import scipy.optimize as opt
import scipy.signal as sig
from itertools import zip_longest, islice, cycle
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from tqdm import tqdm
from numpy.polynomial import polynomial as pln


def gauss(x, s, x0, A):
    """Simple gaussian"""
    return A * np.exp(-(x - x0)**2 / (2 * s**2))


def gauss_cont(x, s, x0, A, k, b):
    """Gaussian plus continuum"""
    return A * np.exp(-(x - x0)**2 / (2 * s**2)) + k * x + b


# FIXME: make rng wider, fit by gauss_cont
def one_peak_fwhm(x, A, wl, spec, guess=1):
    rng = (wl > x - guess) & (wl < x + guess)
    try:
        return 2.355 * np.abs(opt.curve_fit(gauss, wl[rng], spec[rng],
                                            p0=[guess, x, A])[0][0])
    except RuntimeError:
        return 0


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

    fwhm = np.array(list(map(one_peak_fwhm_, wl[peaks], amps)))
    fwhm = np.average(fwhm[fwhm != 0])
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

    # убираем короткие линии
    for n_line in set(y_pred):
        line_y = vectors[:, 1][y_pred == n_line]
        dy = line_y.max() - line_y.min()
        if (dy < (y.max() - y.min()) / 2):
            y_pred[y_pred==n_line] = -1

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

    

    return(vectors[mask], y_pred[mask])


def fine_peak_position_i(row, peak, fwhm=10, x=None):
    if x is None:
        x = np.arange(len(row), dtype='float')
    peak_f = peak
    peak = int(peak)
    fwhm = int(fwhm)

    amp = row[peak]
    b = np.median(row) / 2.
    x_range = slice(max(0, peak - 2 * fwhm), min(len(x), peak + 2 * fwhm))
    x = x[x_range]
    y = row[x_range]
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
        # print(peak)
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


def gauss_spectra(FWHM, peaks, amps, bias=0, step=1, rng=None):
    if rng is None:
        mincoord = peaks.min() - 2 * FWHM
        maxcoord = peaks.max() + 2 * FWHM
    else:
        mincoord = rng[0]
        maxcoord = rng[1]
    coords = np.arange(mincoord, maxcoord, step)
    signal = np.zeros(len(coords)) + bias

    for peak, amp in zip(peaks, amps):
        signal = signal + gauss_fwhm(peak, amp, FWHM, coords)

    return signal, coords


def gauss_fwhm(x0, amp, fwhm, coords):
    sigm = fwhm / 2.3548
    return amp * np.exp(-0.5 * (coords - x0)**2 / (sigm**2))


def get_peaks_h(pos, amp, h=10, d=4, return_h=False):
    FWHM = 2.7
    amps = amp * 100 / amp.max()
    refspec, refcoords = gauss_spectra(FWHM, pos, amps)
    refspec += 1
    k = (refcoords.max() - refcoords.min()) / len(refcoords)
    FWHM_pix_ref = FWHM / k
    peaks_ref_n = find_peaks(refspec, fwhm=FWHM_pix_ref, h=h, d=d)
    peaks_ref = refcoords[peaks_ref_n]
    peaks_ref_theor = find_correspond_peaks(pos, peaks_ref)[0]
    if return_h:
        peaks_ref_h = refspec[peaks_ref_n]
        return(peaks_ref_theor, peaks_ref_n, peaks_ref_h)
    return(peaks_ref_theor, peaks_ref_n)


def find_correspond_peaks(peaks1, peaks2, mask=False):
    # ПРИМЕНЯТЬ ТОЛЬКО КОГДА ПИКИ 1 и 2 БЛИЗКИ ДРУГ К ДРУГУ
    shape = (len(peaks1), 1)
    diff_matrix = np.abs(np.tile(peaks2, shape) - peaks1.reshape(shape))
    # print(diff_matrix.shape)
    mask2 = np.argmin(diff_matrix, axis=1)
    # print(mask2)
    mask1 = np.argmin(diff_matrix, axis=0)
    # print(mask1)

    mask2_ = np.array([], dtype='int')
    for i2, i1 in np.ndenumerate(mask2):
        if mask1[i1] == i2:
            mask2_ = np.append(mask2_, i1)
    mask2_ = list(set(mask2_.tolist()))
    mask2_.sort()

    mask1_ = np.array([], dtype='int')
    for i1, i2 in np.ndenumerate(mask1):
        if mask2[i2] == i1:
            mask1_ = np.append(mask1_, i2)
    mask1_ = list(set(mask1_.tolist()))
    mask1_.sort()

    # элементы из первого массива, на которые ссылается второй
    peaks1_good = np.sort(peaks1[mask1_])
    # print(peaks1_good)
    # элементы из второго массива, на которые ссылается первый
    peaks2_good = np.sort(peaks2[mask2_])
    # print(peaks2_good)
    if mask:
        return(mask1_, mask2_)
    else:
        return(peaks1_good, peaks2_good)


def mynorm(x):
    minmax = [x.min(), x.max()]
    xnorm = (x - x.min()) / (x.max() - x.min())
    return(xnorm, minmax)


def tnorm(x, minmax):
    xnorm = (x - minmax[0]) / (minmax[1] - minmax[0])
    return(xnorm)


def unnorm(x, minmax):
    unorm = x * (minmax[1] - minmax[0]) + minmax[0]
    return unorm


def polyval2d(x, y, k, deg=[2, 2]):
    vandermond = pln.polyvander2d(x, y, deg)
    f = vandermond @ k
    return(f)


def polyfit2d(x, y, f, deg=[2, 2]):
    vandermond = pln.polyvander2d(x, y, deg)
    k = np.linalg.lstsq(vandermond, f)
    return(k)


def sort_labels(clustering, vectors):
    lab = clustering.labels_.astype('int')
    x = vectors[:, 0]
    x_med = []
    enum_lab = set(lab.tolist()) - {-1}
    enum_lab = list(enum_lab)
    for ll in enum_lab:
        x_med.append(np.median(x[lab == ll]))
