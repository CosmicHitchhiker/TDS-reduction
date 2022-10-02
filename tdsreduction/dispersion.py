#! /usr/bin/python3

import numpy as np
import geometry as gm
from matplotlib import pyplot as plt
import bias
import dark
import cosmics
from astropy.io import fits


def get_peaks_clust(neon):
    h = 10  # Во сколько минимально раз пик должен быть выше медианы
    d = 5  # Минимальное расстояние (в fwhm) между пиками

    y, x = np.shape(neon)
    y = np.arange(y)
    x = np.arange(x)

    # За fwhm считаем fwhm (в пикселях) средней (по Y) строки
    fwhm = gm.calc_fwhm(neon[int(len(neon) / 2)])
    print('fwhm = ', fwhm, 'pix')

    # Пики в каждой строчке (list из ndarray разной длины)
    peaks = [gm.find_peaks(row, fwhm=fwhm, h=h, d=d) for row in neon]

    peaks, n_lines = gm.find_lines_cluster(peaks, y, verbose=True)
    return peaks, n_lines, fwhm


def coord_to_lam(image, wlmap, wl):
    if wl is None:
        wl = np.linspace(wlmap[:, 0].max(), wlmap[:, -1].min(), len(wlmap[0]))
    image_res = np.array(list(map(lambda row, spec: np.interp(wl, row, spec),
                                  wlmap, image)))
    return(image_res)


def get_disp_file(data, ref, approx_wl, bias_obj=None, dark_obj=None,
                  cosm_obj=None, hdr=None):
    data_copy = {'data': data.copy()}
    data_copy = bias.process_bias(data_copy, bias_obj)
    if cosm_obj:
        data_copy = cosmics.process_cosmics(data_copy)

    neon_data = data_copy['data'][:, :, ::-1]
    neon = np.sum(neon_data, axis=0)
    peaks, n_lines, fwhm_pix = get_peaks_clust(neon)

    peaks = gm.refine_peaks_i(neon, peaks, fwhm_pix)

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
    plt.figure()
    plt.plot(refspec[1], refspec[0])
    plt.figure()
    plt.plot(neon[225])
    plt.show()

    # k = np.polyfit(np.arange(len(approx_wl))[::], approx_wl, 3)
    # k[-1] += 20
    k2 = [7.41241676e-16, -3.20128023e-12, 5.79035035e-09, -3.04884902e-05,
          9.28102343e-01, 5.62170023e+03]
    approx_line = np.polyval(k2, m_line)
    theor = gm.get_peaks_h(ref[0], ref[1])
    obs_mask, theor_mask = gm.find_correspond_peaks(approx_line,
                                                    theor, mask=True)

    plt.figure()
    plt.plot(approx_line, np.zeros(len(approx_line)), 'o')
    plt.plot(theor, np.ones(len(theor)), 'o')
    plt.plot(approx_line[obs_mask], np.zeros(len(approx_line[obs_mask])), 'o')
    plt.plot(theor[theor_mask], np.ones(len(theor[theor_mask])), 'o')
    plt.show()

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

    deg = [7, 7]
    coeff = gm.polyfit2d(x_fit, y_fit, ref_fit, deg)[0]

    err_fit = []
    # fig = plt.figure()
    deviations = []
    mean = []
    for n, wl in pair_n_wl:
        p = peaks[n_lines == n]
        prediction = gm.polyval2d(gm.tnorm(p[:, 0], x_mm),
                                  gm.tnorm(p[:, 1], y_mm), coeff, deg)
        prediction = gm.unnorm(prediction, ref_mm)
        # print()
        # print(wl)
        # print(np.std(wl-prediction))
        # print(np.mean(wl-prediction))
        mean.append(np.median(wl - prediction))
        err_fit.append(np.std(wl - prediction))
        deviations.append(wl - prediction)
        # plt.figure()
        # plt.plot(p[:, 1], wl - prediction, '.')
        # plt.axhline(0, linestyle='--')
    # plt.show()

    print()
    # print(err_fit)
    print(np.mean(err_fit))

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

    res = fits.PrimaryHDU(WL_map[:, ::-1])
    disp_obj = {'data': WL_map}
    plt.figure()
    plt.imshow(neon)
    plt.show()
    neon_data = {'data': [neon]}
    neon_corrected = process_disp(neon_data, disp_obj)
    print(neon_corrected)
    print()
    print(neon_corrected['data'])
    print()
    print()
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


def disp_from_file(disp_file):
    if isinstance(disp_file, str):
        disp_file = fits.open(disp_file)[0]
    res = {'data': disp_file.data}
    return res


def process_disp(data, disp_obj):
    data_copy = data.copy()
    print(data_copy['data'])
    wlmap = disp_obj['data']
    wl = np.linspace(wlmap[:, 0].max(), wlmap[:, -1].min(), len(wlmap[0]))
    data_copy['data'] = np.array([coord_to_lam(x, wlmap, wl)
                                  for x in data_copy['data']])
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

    disp_file, neon_file = get_disp_file(arc_files, ref, approx_wl, bias_obj,
                                         dark_obj, if_clear_cosmics)
    disp_file.writeto(pargs.out, overwrite=True)
    neon_name = '.'.join((pargs.out).split('.')[:-1]) + '_neon.fits'
    neon_file.writeto(neon_name, overwrite=True)
    return(0)


if __name__ == '__main__':
    import sys
    from utils import open_fits_array_data
    import argparse
    sys.exit(main(sys.argv))
