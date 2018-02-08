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


def bkg(mass_central, ra_central):

    '''
    return a background correction for each central galaxy

    mass_central: mass of the central galaxy
    ra_central: ra value for central galaxy. if ra_central>100, gal in COSMOS, if ra_central<100, gal in XMM-LSS
    '''

    counts_gals_rand = 0
    n = 0

    while n < 15:
        if ra_central > 100:
            ra_rand, dec_rand = 149.6 + random()*1.2, 1.5 + random()*1.3
        else:
            ra_rand, dec_rand = 33.9 + random()*1.4, -5.5 + random()*1.3

        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(ra_rand, dec_rand, unit="deg"), coord_massive_gal, nthneighbor=1)

        if sep2d.degree > 1.0/dis/np.pi*180:
            cat_neighbors_rand = cat_all_z_slice[(cat_all_z_slice['RA'] - ra_rand) ** 2 + (cat_all_z_slice['DEC'] - dec_rand) ** 2 < (0.5 / dis / np.pi * 180) ** 2]
            mass_neighbors_rand = cat_neighbors_rand['MASS_BEST']
            count_gal_rand, edges_rand = np.histogram(10 ** (mass_neighbors_rand - mass_central), np.arange(0, 1.01, 0.1))
            counts_gals_rand += count_gal_rand

            n = n + 1

    return counts_gals_rand/15.


for z in np.arange(1.1, 1.2, 0.1):
    print('============='+str(z)+'================')
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal['ZPHOT']-z)<0.1]  # massive galaxies in this z slice
    cat_all_z_slice = cat_gal[abs(cat_gal['ZPHOT'] - z) < 0.1]  # all galaxies in this z slice

    dis = Planck15.angular_diameter_distance(z).value
    search_r = 0.5/dis/np.pi*180/0.17  # HSC plate scale 0.17 arcsec/pix

    cat_massive_z_slice['RA'].unit = u.deg
    cat_massive_z_slice['DEC'].unit = u.deg
    coord_massive_gal = SkyCoord.guess_from_table(cat_massive_z_slice)

    # search for central galaxies (exclude massive companions)
    sep_angle = 0.0
    mass = cat_massive_z_slice[0]['MASS_BEST']
    id = cat_massive_z_slice[0]['ID']
    for gal in cat_massive_z_slice:
        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(gal['RA'], gal['DEC'], unit="deg"), coord_massive_gal, nthneighbor=2)
        if sep2d.degree < 0.5/dis/np.pi*180:
            if round(sep2d.degree[0], 5) == round(sep_angle, 5):
                if gal['MASS_BEST'] > mass:
                    cat_massive_z_slice = cat_massive_z_slice[cat_massive_z_slice['ID'] != id]
                else:
                    cat_massive_z_slice = cat_massive_z_slice[cat_massive_z_slice['ID'] != gal['ID']]

            sep_angle = sep2d.degree[0]
            mass = gal['MASS_BEST']
            id = gal['ID']

    # counting neighbors for each massive central galaxy
    counts_gals_sf = np.zeros(10)
    counts_gals_q = np.zeros(10)
    counts_sf = 0
    counts_q = 0
    for gal in cat_massive_z_slice:
        cat_all_z_slice = cat_all_z_slice[cat_all_z_slice['ID'] != gal['ID']]
        cat_neighbors = cat_all_z_slice[(cat_all_z_slice['RA']-gal['RA'])**2+(cat_all_z_slice['DEC']-gal['DEC'])**2 < (0.5/dis/np.pi*180)**2]

        cat_neighbors_sf = cat_neighbors[cat_neighbors['SSFR_BEST'] > -11]
        cat_neighbors_q = cat_neighbors[cat_neighbors['SSFR_BEST'] < -11]

        mass_neighbors = cat_neighbors['MASS_BEST']
        mass_central = gal['MASS_BEST']


        if gal['SSFR_BEST'] > -11:  # star forming galaxies
            count_gal_sf, edges = np.histogram(10**(mass_neighbors-mass_central), np.arange(0, 1.01, 0.1))
            count_gal_sf = np.array(count_gal_sf, dtype='float64')
            count_gal_sf -= bkg(mass_central, gal['RA']) # background/foreground correction

            counts_gals_sf += count_gal_sf
            counts_sf += 1

        else:  # quiescent galaxies
            count_gal_q, edges = np.histogram(10 ** (mass_neighbors - mass_central), np.arange(0, 1.01, 0.1))
            count_gal_q = np.array(count_gal_q, dtype='float64')
            count_gal_q -= bkg(mass_central, gal['RA'])  # background/foreground correction

            counts_gals_q += count_gal_q
            counts_q += 1

    fig = plt.figure(figsize=(7, 6))
    plt.rc('font', family='serif'), plt.rc('xtick', labelsize=15), plt.rc('ytick', labelsize=15)

    plt.errorbar(x=np.arange(0, 0.91, 0.1)+0.05, y=counts_gals_sf/counts_sf,
                 yerr=np.sqrt(counts_gals_sf)/counts_sf, fmt='ob', markersize=12,
                 capsize=3, elinewidth=1, label='star-forming')
    plt.errorbar(x=np.arange(0, 0.91, 0.1) + 0.05, y=counts_gals_q/counts_q,
                 yerr=np.sqrt(counts_gals_q)/counts_q, fmt='^r', markersize=12,
                 capsize=3, elinewidth=1, label='quiescent')

    plt.ylabel('neighbor counts', fontsize=15)
    plt.xlabel(r'$M_{sat}/M_{central}$', fontsize=16)
    plt.text(0.1, 0.007, 'z='+str(z-0.1)+'~'+str(z+0.1), fontsize=14)
    plt.legend(loc='upper right', fontsize=15)

    plt.yscale('log')
    plt.show()