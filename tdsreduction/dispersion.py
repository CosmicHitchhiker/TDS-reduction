#! /usr/bin/python3

import numpy as np
import geometry as gm
from matplotlib import pyplot as plt
import bias
import dark
import cosmics
import corrections
from astropy.io import fits
import argparse
from matplotlib.patches import ConnectionPatch


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


def coord_to_lam(image, wlmap, wl):
    if wl is None:
        wl = np.linspace(wlmap[:, 0].max(), wlmap[:, -1].min(), len(wlmap[0]))
    image_res = np.array(list(map(lambda row, spec: np.interp(wl, row, spec),
                                  wlmap, image)))
    return(image_res)


def calc_subplot_dim(n):
    m = round(n**0.5)
    if m**2 < n:
        return m, m + 1
    return m, m


def get_dispersion_file(data, ref, approx_wl, bias_obj=None, dark_obj=None,
                  cosm_obj=None, hdr=None):
    data_copy = {'data': data.copy()}
    data_copy = bias.process_bias(data_copy, bias_obj)
    if cosm_obj:
        data_copy = cosmics.process_cosmics(data_copy)

    if hdr['DISP'] == 'R':
        neon_data = data_copy['data'][:, :, ::-1]
    else:
        neon_data = data_copy['data']
    neon = np.sum(neon_data, axis=0)

    peaks, n_lines, fwhm_pix = get_peaks_clust_setup(neon)

    m_line = []

    n_ordered = np.array(sorted(set(n_lines)))

    for n in n_ordered:
        # ПОДУМАТЬ МОЖНО ЛИ МЕДИАНУ
        m_line.append(np.median(peaks[n_lines == n][:, 0]))

    m_line = np.array(m_line)

    n_ordered = n_ordered[np.argsort(m_line)]
    m_line = np.sort(m_line)
    plt.figure()
    plt.imshow(neon)
    plt.plot(m_line, np.ones(len(m_line)) * 225, 'ro')
    plt.show()

    refspec = gm.gauss_spectra(fwhm_pix, ref[0], ref[1], bias=0,
                               step=1, rng=None)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(refspec[1], refspec[0])
    ax[0].set_title('Reference')
    ax[1].plot(neon[225])
    ax[1].set_title('Observed')
    plt.show()

    # k = np.polyfit(np.arange(len(approx_wl))[::], approx_wl, 3)
    # k[-1] += 20
    if hdr['DISP'] == 'R':
        k2 = [7.41241676e-16, -3.20128023e-12, 5.79035035e-09, -3.04884902e-05,
              9.28102343e-01, 5.62170023e+03]
    if hdr['DISP'] == 'B':
        k2 = [6.14095880e-15, -3.71267430e-11, 1.01336659e-07, -1.75917785e-04,
              1.35359449e+00, 3.32310482e+03]
    approx_line = np.polyval(k2, m_line)
    theor, theor_n = gm.get_peaks_h(ref[0], ref[1])
    obs_mask, theor_mask = gm.find_correspond_peaks(approx_line,
                                                    theor, mask=True)

    fig, ax = plt.subplots(2, 1)
    m_line_n = m_line.astype(int)
    ax[0].plot(refspec[1], refspec[0])
    ax[0].plot(refspec[1][theor_n], refspec[0][theor_n], 'o', ms=3)
    ax[1].plot(neon[225])
    ax[1].plot(m_line_n, neon_data[0, 225][m_line_n], 'o', ms=3)

    for i in range(len(approx_line[obs_mask])):
        xA = refspec[1][theor_n][theor_mask][i]
        yA = refspec[0][theor_n][theor_mask][i]
        xB = m_line.astype(int)[obs_mask][i]
        yB = neon_data[0, 225][m_line.astype(int)][obs_mask][i]
        con = ConnectionPatch((xA, yA), (xB, yB), 'data', 'data', axesA=ax[0],
                              axesB=ax[1], lw=0.5)
        fig.add_artist(con)
    plt.show()

    peaks = gm.refine_peaks_i(neon, peaks, fwhm_pix)

    ref_peaks = np.zeros(len(peaks))

    pair_n_wl = np.array([n_ordered[obs_mask], theor[theor_mask]]).T

    for n, wl in pair_n_wl:
        ref_peaks[n_lines == n] = wl

    mask = (ref_peaks != 0)

    x_peaks = peaks[:, 0]
    y_peaks = peaks[:, 1]

    x_fit, x_mm = gm.mynorm(x_peaks[mask])
    y_fit, y_mm = gm.mynorm(y_peaks[mask])
    ref_fit, ref_mm = gm.mynorm(ref_peaks[mask])

    deg = [5, 5]
    coeff = gm.polyfit2d(x_fit, y_fit, ref_fit, deg)[0]

    err_fit = []
    # fig = plt.figure()
    deviations = []
    mean = []

    dim_subpl = calc_subplot_dim(len(pair_n_wl))
    fig, ax = plt.subplots(*dim_subpl)
    i = 0
    j = 0
    for n, wl in pair_n_wl:
        p = peaks[n_lines == n]
        prediction = gm.polyval2d(gm.tnorm(p[:, 0], x_mm),
                                  gm.tnorm(p[:, 1], y_mm), coeff, deg)
        prediction = gm.unnorm(prediction, ref_mm)
        mean.append(np.median(wl - prediction))
        err_fit.append(np.std(wl - prediction))
        deviations.append(wl - prediction)
        ax[j, i].plot(p[:, 1], wl - prediction, '.')
        ax[j, i].axhline(0, linestyle='--', label=str(wl))
        ax[j, i].legend()
        i += 1
        if i == dim_subpl[0]:
            i = 0
            j += 1
    fig.show()
    print()
    # print(err_fit)
    print(np.mean(err_fit))
    print(np.mean(err_fit) * 3e+5 / 5500.)

    plt.figure()
    plt.ylim(-0.12, 0.12)
    plt.errorbar(pair_n_wl[:, 1], mean, yerr=err_fit, linestyle='', marker='.')
    plt.show()

    y = np.arange(len(neon))
    x = np.arange(len(neon[0]))
    # grid = np.dstack(np.meshgrid(x, y, indexing='ij'))
    xx, yy = np.meshgrid(x, y)

    WL_map = gm.polyval2d(gm.tnorm(xx, x_mm), gm.tnorm(yy, y_mm), coeff, deg)
    WL_map = gm.unnorm(WL_map, ref_mm)

    # if hdr['DISP'] == 'R':
    #     res = fits.PrimaryHDU(WL_map[:, ::-1])
    # else:
    #     res = fits.PrimaryHDU(WL_map[:])
    res = fits.PrimaryHDU(WL_map[:])
    disp_obj = {'data': res.data}
    neon_data = {'data': np.array([neon[:,::-1]])}
    # neon_data = data_copy
    neon_corrected = process_dispersion(neon_data, disp_obj)

    plt.figure()
    plt.imshow(neon_data['data'][0])
    plt.title('Neon')
    plt.figure()
    plt.imshow(neon_corrected['data'][0])
    plt.title('neon_corrected')
    plt.show()

    if hdr is not None:
        neon_res = fits.PrimaryHDU(neon_corrected['data'][0], header=hdr)
    else:
        neon_res = fits.PrimaryHDU(neon_corrected['data'][0])
    neon_res.header['CRPIX1'] = 1
    crval = round(neon_corrected['wl'][0], 3)
    crdelt = round((neon_corrected['wl'][1] - neon_corrected['wl'][0]), 3)
    neon_res.header['CRVAL1'] = crval
    neon_res.header['CDELT1'] = crdelt
    neon_res.header['CTYPE1'] = 'WAVE'
    return res, neon_res


def dispersion_from_file(disp_file):
    if isinstance(disp_file, str):
        disp_file = fits.open(disp_file)[0]
    res = {'data': disp_file.data}
    return res


def process_dispersion(data, disp_obj):
    data_copy = data.copy()
    if disp_obj is None:
        return data_copy
    wlmap = disp_obj['data']
    wl1 = wlmap[:, 0].max()
    wl2 = wlmap[:, -1].min()
    wl = np.linspace(wl1, wl2, len(wlmap[0]))
    print(wl)

    if wl2 > 7000:
        data_copy['data'] = data_copy['data'][:, :, ::-1]

    data_copy['data'] = np.array([coord_to_lam(x, wlmap, wl)
                                  for x in data_copy['data']])
    # if wl1 < wl2:
    #     data_copy['data'] = np.array([coord_to_lam(x, wlmap, wl)
    #                                   for x in data_copy['data']])
    # else:
    #     wl = wl[::-1]
    #     data_copy['data'] = data_copy['data'][:, :, ::-1]
    #     data_copy['data'] = np.array([coord_to_lam(x, wlmap, wl)
    #                                   for x in data_copy['data']])
    data_copy['wl'] = wl
    return data_copy


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+',
                        help="""fits files with neon frames and one .dat file
                        with reference spectrum""")
    parser.add_argument('-d', '--dir', help="directory with input files")
    parser.add_argument('-o', '--out', default='../data/disp_map.fits',
                        help='output file')
    parser.add_argument('-G', '--GEOMETRY', help="file with correction map")
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

    if pargs.GEOMETRY:
        corr_obj = corrections.corrections_from_file(pargs.GEOMETRY)
    else:
        corr_obj = None

    file_names = pargs.filenames
    arc_names = []
    ref_name = None
    approx_name = None
    for name in file_names:
        ending = name.split('.')[-1]
        if ending == "fit":
            approx_name = name
        elif ending == "txt":
            ref_name = name
        elif ending == "fits":
            arc_names.append(name)

    print()
    print("Approx ", approx_name)
    print("Ref ", ref_name)
    print("Arc ", arc_names)

    if pargs.dir:
        arc_names = [pargs.dir + x for x in arc_names]
    arc_files, headers = open_fits_array_data(arc_names, header=True)

    hm = fits.open(approx_name)
    approx_wl = hm['wave'].data[250]

    ref = np.loadtxt(ref_name).T
    ref[1] = ref[1][np.argsort(ref[0])]
    ref[0] = np.sort(ref[0])

    disp_file, neon_file = get_dispersion_file(arc_files, ref, approx_wl, bias_obj,
                                         dark_obj, if_clear_cosmics,
                                         headers[0])
    disp_file.writeto(pargs.out, overwrite=True)
    neon_name = '.'.join((pargs.out).split('.')[:-1]) + '_neon.fits'
    neon_file.writeto(neon_name, overwrite=True)
    return(0)


if __name__ == '__main__':
    import sys
    from utils import open_fits_array_data
    sys.exit(main(sys.argv))
