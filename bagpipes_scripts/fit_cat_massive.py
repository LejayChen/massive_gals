import numpy as np
import bagpipes as pipes
from astropy.table import Table
import sys


def Jy2ab(flux):
    return 2.5 * (23 - np.log10(flux)) - 48.6


def ab2Jy(mag):
    return 10 ** (23 - (mag + 48.6) / 2.5)


def aberr2Jy(mag, magerr):
    return 5811.33 * np.e ** (-0.921034 * mag) * magerr


def load_gal(ID):
    # cat = Table.read('../catalogs/deep2_photoz_v2.fits')
    gal = cat[cat['ID'].astype(str) == str(ID)]

    # load magnitudes and magerrs
    u, g, r, i, z, y = gal['MAG_APER_2s_u'], gal['MAG_APER_2s_g'], gal['MAG_APER_2s_r'], gal['MAG_APER_2s_i'], gal[
        'MAG_APER_2s_z'], gal['MAG_APER_2s_y']
    uerr, gerr, rerr, ierr, zerr, yerr = gal['MAGERR_APER_2s_u'], gal['MAGERR_APER_2s_g'], gal['MAGERR_APER_2s_r'], gal[
        'MAGERR_APER_2s_i'], gal['MAGERR_APER_2s_z'], gal['MAGERR_APER_2s_y']
    mags = np.array([u, g, r, i, z, y])
    magerrs = np.array([uerr, gerr, rerr, ierr, zerr, yerr])

    # mag correction (Sawicki & Yee 1997)
    i_corr = gal['i'] - gal['MAG_APER_2s_i']
    mags = mags + i_corr
    magerrs = np.sqrt(gal['i_err'] ** 2 + magerrs ** 2)

    # mag to flux conversion
    fluxes = ab2Jy(mags)
    fluxerrs = aberr2Jy(mags, magerrs)
    photometry = np.c_[fluxes, fluxerrs]

    # NaNs and 0s, Jy to mJy
    for i in range(len(photometry)):
        photometry[i, 0] = photometry[i, 0] * 1e6  # Jy to mJy (flux)
        photometry[i, 1] = photometry[i, 1] * 1e6  # Jy to mJy (fluxerr)
        if (photometry[i, 0] <= 0.) or (photometry[i, 1] <= 0):  # deal with non detections
            photometry[i, :] = [0., 9.9 * 10 ** 9.]

    # Enforce a maximum SNR of 20.
    for i in range(len(photometry)):
        max_snr = 20.

        if photometry[i, 0] / photometry[i, 1] > max_snr:
            photometry[i, 1] = photometry[i, 0] / max_snr

    return photometry


def load_gal_model_cat(ID):
    gal = cat[cat['ID'].astype(str) == str(ID)]

    # mag to flux conversion
    fluxes = [gal['flux_u'], gal['flux_g'], gal['flux_r'], gal['flux_i'], gal['flux_z'], gal['flux_y']]
    fluxerrs = [gal['fluxerr_u'], gal['fluxerr_g'], gal['fluxerr_r'], gal['fluxerr_i'], gal['fluxerr_z'], gal['fluxerr_y']]
    photometry = np.c_[fluxes, fluxerrs]

    # NaNs and 0s, Jy to mJy
    for i in range(len(photometry)):
        photometry[i, 0] = photometry[i, 0] * 1e6  # Jy to mJy (flux)
        photometry[i, 1] = photometry[i, 1] * 1e6  # Jy to mJy (fluxerr)
        if (photometry[i, 0] <= 0.) or (photometry[i, 1] <= 0):  # deal with non detections
            photometry[i, :] = [0., 9.9 * 10 ** 9.]

    # Enforce a maximum SNR of 20.
    for i in range(len(photometry)):
        max_snr = 20.

        if photometry[i, 0] / photometry[i, 1] > max_snr:
            photometry[i, 1] = photometry[i, 0] / max_snr

    return photometry


# filter curves (not including atmospheric absoption)
curve_u = '../filters_new/CFHT_Megaprime.u.txt'
curve_g = '../filters_new/Subaru_HSC.g.txt'
curve_r = '../filters_new/Subaru_HSC.r.txt'
curve_i = '../filters_new/Subaru_HSC.i.txt'
curve_z = '../filters_new/Subaru_HSC.z.txt'
curve_y = '../filters_new/Subaru_HSC.Y.txt'
filt_list_new = [curve_u, curve_g, curve_r, curve_i, curve_z, curve_y]

# filter curves (not including atmospheric absoption)
curve_u_full = '../hsc_responses_all_rev3/U.MP9302.txt'
curve_g_full = '../hsc_responses_all_rev3/hsc_g_v2018.dat'
curve_r_full = '../hsc_responses_all_rev3/hsc_r_v2018.dat'
curve_i_full = '../hsc_responses_all_rev3/hsc_i_v2018.dat'
curve_z_full = '../hsc_responses_all_rev3/hsc_z_v2018.dat'
curve_y_full = '../hsc_responses_all_rev3/hsc_y_v2018.dat'
filt_list_full = [curve_u_full, curve_g_full, curve_r_full, curve_i_full, curve_z_full, curve_y_full]

# fit instructions
delayed = {}
delayed["age"] = (0.1, 14.)                 # time since first star forming
delayed["tau"] = (0.1, 10.0)                # timescale of star forming
delayed["massformed"] = (6., 14.)
delayed["metallicity"] = (0., 2.5)

dust = {}                                 # Dust component
dust["type"] = "Calzetti"                 # Define the shape of the attenuation curve
dust["Av"] = (0., 2.)                     # Vary Av between 0 and 2 magnitudes

nebular = {}
nebular["logU"] = -2          # Log_10 of the ionization parameter.

fit_instructions = {}
fit_instructions["redshift"] = (0., 1.5)
fit_instructions["delayed"] = delayed
fit_instructions["dust"] = dust
fit_instructions["nebular"] = nebular

# load the catalog
cat_name = sys.argv[2]
if cat_name in ['cosmos', 'xmm', 'deep', 'elais']:
    cat = Table.read('../catalogs/'+cat_name+'_to_fit_'+sys.argv[3]+'.fits')
    IDs = cat['ID']
    redshifts = cat['Z_BEST']
    redshift_sigmas = np.maximum(cat['Z_BEST'] - cat['Z_BEST68_LOW'],
                                 (cat['Z_BEST68_HIGH'] - cat['Z_BEST68_LOW']) / 2).tolist()
    fit_cat = pipes.fit_catalogue(IDs, fit_instructions, load_gal, spectrum_exists=False,
                                  cat_filt_list=filt_list_full, run=sys.argv[1],
                                  redshifts=redshifts, redshift_sigma=redshift_sigmas)
else:
    cat = Table.read('../catalogs/' + cat_name)
    IDs = cat['ID']
    redshifts = cat['z']
    redshift_sigmas = cat['z_sigma'].tolist()
    fit_cat = pipes.fit_catalogue(IDs, fit_instructions, load_gal_model_cat, spectrum_exists=False,
                                  cat_filt_list=filt_list_full, run=sys.argv[1],
                                  redshifts=redshifts, redshift_sigma=redshift_sigmas)

# fitting
fit_cat.fit(verbose=True, mpi_serial=True, n_live=100)
