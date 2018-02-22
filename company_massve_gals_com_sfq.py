from astropy.table import Table
from astropy.cosmology import Planck15
from astropy.coordinates import SkyCoord, match_coordinates_sky
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from random import random

cat = Table.read('CUT_CLAUDS_HSC_VISTA_Ks23.3_PHYSPARAM_TM.fits')
cat_gal = cat[cat['CLASS'] == 0]
cat_massive_gal = cat_gal[cat_gal['MASS_BEST'] > 11.3]


def bkg(mass_central, ra_central, cat_all_z_slice, act='sf'):
    '''
    return a background correction for each central galaxy

    mass_central: mass of the central galaxy
    ra_central: ra value for central galaxy. if ra_central>100, gal in COSMOS, if ra_central<100, gal in XMM-LSS
    '''

    counts_gals_rand = 0
    n = 0

    while n < 20:
        same_field = False
        while same_field == False:
            id_rand = int(random()*len(cat_all_z_slice))
            ra_rand = cat_all_z_slice[id_rand]['RA']
            dec_rand = cat_all_z_slice[id_rand]['DEC']

            if ra_rand > 100 and ra_central > 100:
                same_field = True
            elif ra_rand < 100 and ra_central < 100:
                same_field = True
            else:
                same_field = False

        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(ra_rand, dec_rand, unit="deg"), coord_massive_gal,
                                                   nthneighbor=1)

        if sep2d.degree > 1.0 / dis / np.pi * 180:
            cat_neighbors_rand = cat_all_z_slice[
                (cat_all_z_slice['RA'] - ra_rand) ** 2 + (cat_all_z_slice['DEC'] - dec_rand) ** 2 < (0.5 / dis / np.pi * 180) ** 2]

            if act == 'sf':
                cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['SSFR_BEST'] > -11]
            else:
                cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['SSFR_BEST'] < -11]

            mass_neighbors_rand = cat_neighbors_rand['MASS_BEST']
            count_gal_rand, edges_rand = np.histogram(10 ** (mass_neighbors_rand - mass_central),
                                                      np.arange(0, 1.01, 0.1))
            counts_gals_rand += count_gal_rand

            n = n + 1

    return counts_gals_rand / 20.


for z in np.arange(0.5, 2.1, 0.5):
    print('=============' + str(z) + '================')
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal['ZPHOT'] - z) < 0.1]  # massive galaxies in this z slice
    cat_all_z_slice = cat_gal[abs(cat_gal['ZPHOT'] - z) < 0.1]  # all galaxies in this z slice

    dis = Planck15.angular_diameter_distance(z).value
    search_r = 0.5 / dis / np.pi * 180 / 0.17  # HSC plate scale 0.17 arcsec/pix

    cat_massive_z_slice['RA'].unit = u.deg
    cat_massive_z_slice['DEC'].unit = u.deg
    coord_massive_gal = SkyCoord.guess_from_table(cat_massive_z_slice)

    # search for central galaxies (exclude massive companions)
    sep_angle = 0.0
    mass = cat_massive_z_slice[0]['MASS_BEST']
    id = cat_massive_z_slice[0]['ID']
    for gal in cat_massive_z_slice:
        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(gal['RA'], gal['DEC'], unit="deg"), coord_massive_gal,
                                                   nthneighbor=2)
        if sep2d.degree < 0.5 / dis / np.pi * 180:
            if round(sep2d.degree[0], 5) == round(sep_angle, 5):
                if gal['MASS_BEST'] > mass:
                    cat_massive_z_slice = cat_massive_z_slice[cat_massive_z_slice['ID'] != id]
                else:
                    cat_massive_z_slice = cat_massive_z_slice[cat_massive_z_slice['ID'] != gal['ID']]

            sep_angle = sep2d.degree[0]
            mass = gal['MASS_BEST']
            id = gal['ID']

    # counting neighbors for each massive central galaxy
    counts_gals_sf_sf = np.zeros(10)
    counts_gals_sf_q = np.zeros(10)
    counts_gals_q_q = np.zeros(10)
    counts_gals_q_sf = np.zeros(10)
    counts_sf_sf = 0
    counts_sf_q = 0
    counts_q_q = 0
    counts_q_sf = 0

    for gal in cat_massive_z_slice:
        cat_all_z_slice = cat_all_z_slice[cat_all_z_slice['ID'] != gal['ID']]
        cat_neighbors = cat_all_z_slice[
            (cat_all_z_slice['RA'] - gal['RA']) ** 2 + (cat_all_z_slice['DEC'] - gal['DEC']) ** 2 < (
            0.5 / dis / np.pi * 180) ** 2]

        cat_neighbors_sf = cat_neighbors[cat_neighbors['SSFR_BEST'] > -11]
        cat_neighbors_q = cat_neighbors[cat_neighbors['SSFR_BEST'] < -11]

        mass_neighbors_sf = cat_neighbors_sf['MASS_BEST']
        mass_neighbors_q = cat_neighbors_q['MASS_BEST']
        mass_central = gal['MASS_BEST']

        if gal['SSFR_BEST'] > -11:  # star forming central galaxies
            # star-forming satellite galaxies
            count_gal_sf_sf, edges = np.histogram(10 ** (mass_neighbors_sf - mass_central), np.arange(0, 1.01, 0.1))
            count_gal_sf_sf = np.array(count_gal_sf_sf, dtype='float64')
            count_gal_sf_sf -= bkg(mass_central, gal['RA'], cat_all_z_slice, act='sf')  # background/foreground correction
            counts_gals_sf_sf += count_gal_sf_sf
            counts_sf_sf += 1

            # quiescent satellite galaxies
            count_gal_sf_q, edges = np.histogram(10 ** (mass_neighbors_q - mass_central), np.arange(0, 1.01, 0.1))
            count_gal_sf_q = np.array(count_gal_sf_q, dtype='float64')
            count_gal_sf_q -= bkg(mass_central, gal['RA'], cat_all_z_slice,  act='q')  # background/foreground correction
            counts_gals_sf_q += count_gal_sf_q
            counts_sf_q += 1

        else:  # quiescent central galaxies
            # quiescent satellite galaxies
            count_gal_q_q, edges = np.histogram(10 ** (mass_neighbors_q - mass_central), np.arange(0, 1.01, 0.1))
            count_gal_q_q = np.array(count_gal_q_q, dtype='float64')
            count_gal_q_q -= bkg(mass_central, gal['RA'], cat_all_z_slice,  act='q')  # background/foreground correction
            counts_gals_q_q += count_gal_q_q
            counts_q_q += 1

            # star-forming satellite galaxies
            count_gal_q_sf, edges = np.histogram(10 ** (mass_neighbors_sf - mass_central), np.arange(0, 1.01, 0.1))
            count_gal_q_sf = np.array(count_gal_q_sf, dtype='float64')
            count_gal_q_sf -= bkg(mass_central, gal['RA'], cat_all_z_slice,  act='sf')  # background/foreground correction
            counts_gals_q_sf += count_gal_q_sf
            counts_q_sf += 1

    fig = plt.figure(figsize=(7, 6))
    plt.rc('font', family='serif'), plt.rc('xtick', labelsize=15), plt.rc('ytick', labelsize=15)

    #  with star-forming central galaxies
    plt.errorbar(x=np.arange(0, 0.91, 0.1) + 0.05, y=counts_gals_sf_sf / counts_sf_sf,
                 yerr=np.sqrt(counts_gals_sf_sf) / counts_sf_sf, fmt='ob', markersize=12,
                 capsize=3, elinewidth=1, label='sf-sf')  # sf satellites
    plt.errorbar(x=np.arange(0, 0.91, 0.1) + 0.05, y=counts_gals_sf_q / counts_sf_q,
                 yerr=np.sqrt(counts_gals_sf_q) / counts_sf_q, fmt='or', markersize=12,
                 capsize=3, elinewidth=1, label='sf-q')  # q satellites

    #  with quiescent central galaxies
    plt.errorbar(x=np.arange(0, 0.91, 0.1) + 0.05, y=counts_gals_q_sf / counts_q_sf,
                 yerr=np.sqrt(counts_gals_q_sf) / counts_q_sf, fmt='^b', markersize=12,
                 capsize=3, elinewidth=1, label='q-sf', markerfacecolor="None")  # sf satellites
    plt.errorbar(x=np.arange(0, 0.91, 0.1) + 0.05, y=counts_gals_q_q / counts_q_q,
                 yerr=np.sqrt(counts_gals_q_q) / counts_q_q, fmt='^r', markersize=12,
                 capsize=3, elinewidth=1, label='q-q', markerfacecolor="None")  # q satellites

    plt.ylabel('neighbor counts', fontsize=15)
    plt.xlabel(r'$M_{sat}/M_{central}$', fontsize=16)
    plt.annotate('z=' + str(z - 0.1) + '~' + str(z + 0.1), (10,10), fontsize=14, xycoords='axes points')
    plt.legend(loc='upper right', fontsize=15)

    plt.yscale('log')
    plt.savefig('companion_counts_plots_sf_q/N_sat_cor_sep_'+str(z)+'.png')