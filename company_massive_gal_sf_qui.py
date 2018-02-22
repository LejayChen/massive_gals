from astropy.table import Table
from astropy.cosmology import Planck15
from astropy.coordinates import SkyCoord, match_coordinates_sky
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from random import random

cat = Table.read('CUT2_CLAUDS_HSC_VISTA_Ks23.3_PHYSPARAM_TM.fits')
cat_gal = cat[cat['CLASS'] == 0]
cat_massive_gal = cat_gal[cat_gal['MASS_BEST'] > 11.3]


def bkg(mass_central, ra_central, cat_all_z_slice_rand, coord_massive_gal_rand):

    '''
    return a background correction for each central galaxy
    mass_central: mass of the central galaxy, ra_central: ra value for central galaxy. if ra_central>100, gal in COSMOS, if ra_central<100, gal in XMM-LSS
    '''

    counts_gals_rand = 0
    n = 0
    num_p = 15

    while n < num_p:  # get several blank pointings to estimate background
        same_field = False  # find a random pointing in the same field as the central galaxy
        while not same_field:
            id_rand = int(random()*len(cat_all_z_slice_rand))
            ra_rand = cat_all_z_slice[id_rand]['RA'] + random()*1.0/dis/np.pi*180
            dec_rand = cat_all_z_slice[id_rand]['DEC'] + random()*1.0/dis/np.pi*180
            if ra_rand > 100 and ra_central > 100:
                same_field = True
            elif ra_rand < 100 and ra_central < 100:
                same_field = True
            else:
                same_field = False

        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(ra_rand, dec_rand, unit="deg"), coord_massive_gal_rand, nthneighbor=1)
        if sep2d.degree > 2.0/dis/np.pi*180:
            coord_rand = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)
            cat_neighbors_rand = cat_all_z_slice[coord_all_z_slice.separation(coord_rand).degree < 0.5/dis/np.pi*180]
            if len(cat_neighbors_rand)==0: continue

            mass_neighbors_rand = cat_neighbors_rand['MASS_BEST']
            count_gal_rand, edges_rand = np.histogram(10 ** (mass_neighbors_rand - mass_central), np.arange(0, 1.01, 0.1))
            counts_gals_rand += count_gal_rand

            n = n + 1

    return counts_gals_rand/float(num_p)


for z in np.arange(1.5, 2.0, 0.1):
    print('============='+str(z)+'================')
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal['ZPHOT']-z)<0.1]  # massive galaxies in this z slice
    cat_all_z_slice = cat_gal[abs(cat_gal['ZPHOT'] - z) < 0.1]  # all galaxies in this z slice

    dis = Planck15.angular_diameter_distance(z).value
    search_r = 0.5/dis/np.pi*180/0.17  # HSC plate scale 0.17 arcsec/pix

    cat_massive_z_slice['RA'].unit = u.deg
    cat_massive_z_slice['DEC'].unit = u.deg
    coord_massive_gal = SkyCoord.guess_from_table(cat_massive_z_slice)

    # counting neighbors for each massive central galaxy
    counts_gals_sf = np.zeros(10)
    counts_gals_q = np.zeros(10)
    counts_sf = 0
    counts_q = 0
    for gal in cat_massive_z_slice:
        coord_all_z_slice = SkyCoord(cat_all_z_slice['RA'] * u.deg, cat_all_z_slice['DEC'] * u.deg)
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)
        cat_neighbors = cat_all_z_slice[coord_all_z_slice.separation(coord_gal).degree < 0.5/dis/np.pi*180]
        cat_neighbors = cat_neighbors[cat_neighbors['ID'] != gal['ID']]
        cat_neighbors_sf = cat_neighbors[cat_neighbors['SSFR_BEST'] > -11]
        cat_neighbors_q = cat_neighbors[cat_neighbors['SSFR_BEST'] < -11]

        mass_neighbors = cat_neighbors['MASS_BEST']
        mass_central = gal['MASS_BEST']

        if len(cat_neighbors) == 0:  # exlucde central gals which has no companion
            continue

        if gal['MASS_BEST'] < max(cat_neighbors['MASS_BEST']):  # exclude central gals which has larger mass companion
            continue

        if gal['SSFR_BEST'] > -11:  # star forming central galaxies
            count_gal_sf, edges = np.histogram(10**(mass_neighbors - mass_central), np.arange(0, 1.01, 0.1))
            count_gal_sf = np.array(count_gal_sf, dtype='float64')
            count_gal_sf -= bkg(mass_central, gal['RA'], cat_all_z_slice, coord_massive_gal)  # background/foreground correction

            counts_gals_sf += count_gal_sf
            counts_sf += 1

        else:  # quiescent central galaxies
            count_gal_q, edges = np.histogram(10**(mass_neighbors - mass_central), np.arange(0, 1.01, 0.1))
            count_gal_q = np.array(count_gal_q, dtype='float64')
            count_gal_q -= bkg(mass_central, gal['RA'], cat_all_z_slice, coord_massive_gal)  # background/foreground correction

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
    plt.ylim([0.001, 5])
    plt.savefig('companion_counts_gal_sfq/'+str(z)+'.png')