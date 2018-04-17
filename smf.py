from astropy.table import Table
from astropy.cosmology import WMAP9
from astropy.coordinates import SkyCoord, match_coordinates_sky
from random import random
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

cat = Table.read('CUT2_CLAUDS_HSC_VISTA_Ks23.3_PHYSPARAM_TM.fits')
cat_gal = cat[cat['CLASS'] == 0]
cat_massive_gal = cat_gal[cat_gal['MASS_BEST'] > 11.15]


def bkg(ra_central, cat_all_z_slice_rand, coord_massive_gal_rand):
    '''
    return a background correction for each central galaxy by looking at blank pointings without massive galaxies.
    mass_central: mass of the central galaxy,  ra_central: ra value for central galaxy. if ra_central>100, gal in COSMOS, if ra_central<100, gal in XMM-LSS
    '''
    n = 0
    num_p = 5  # number of blank pointing's
    while n < num_p:  # get several blank pointing's to estimate background
        same_field = False  # find a random pointing in the same field as the central galaxy
        while not same_field:
            id_rand = int(random()*len(cat_all_z_slice_rand))
            ra_rand = cat_all_z_slice_rand[id_rand]['RA'] + random()*2.0/dis/np.pi*180
            dec_rand = cat_all_z_slice_rand[id_rand]['DEC'] + random()*2.0/dis/np.pi*180
            if ra_rand > 100 and ra_central > 100:
                same_field = True
            elif ra_rand < 100 and ra_central < 100:
                same_field = True
            else:
                same_field = False

        coord_rand = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)
        idx, sep2d, dist3d = match_coordinates_sky(coord_rand, coord_massive_gal_rand, nthneighbor=1)
        if sep2d.degree > 2.1/dis/np.pi*180:  # make sure the random pointing is away from any central galaxy (blank)
            cat_neighbors_rand = cat_all_z_slice_rand[abs(cat_all_z_slice_rand['RA'] - ra_rand) < 1.0 / dis / np.pi * 180]
            cat_neighbors_rand = cat_neighbors_rand[abs(cat_neighbors_rand['DEC'] - dec_rand) < 1.0 / dis / np.pi * 180]
            coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)
            cat_neighbors_rand = cat_neighbors_rand[coord_neighbors_rand.separation(coord_rand).degree < 1.0 / dis / np.pi * 180]
            if len(cat_neighbors_rand) == 0: continue  # no gals (in masked region)

            mass_neighbors_rand = cat_neighbors_rand['MASS_BEST']
            n = n + 1

    return mass_neighbors_rand

bin_counts = 60
for z in np.arange(7, 7.1, 1)/10.:
    print('============='+str(z)+'================')
    dis = WMAP9.angular_diameter_distance(z).value  # angular diameter distance at redshift z
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal['ZPHOT'] - z) < 0.2]  # massive galaxies in this z slice
    cat_all_z_slice = cat_gal[abs(cat_gal['ZPHOT'] - z) < 0.3]  # all galaxies in this z slice
    coord_massive_gal = SkyCoord(cat_massive_z_slice['RA'] * u.deg, cat_massive_z_slice['DEC'] * u.deg)

    massive_counts = len(cat_massive_z_slice)
    SMF_z = np.zeros(bin_counts)
    SMF_z_bkg = np.zeros(bin_counts)
    for gal in cat_massive_z_slice:
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)
        cat_neighbors_z_slice = cat_all_z_slice #[abs(cat_all_z_slice['ZPHOT'] - gal['ZPHOT']) < 1.5 * 0.03 * (1 + z)]
        cat_neighbors = cat_neighbors_z_slice[abs(cat_neighbors_z_slice['RA'] - gal['RA']) < 1.0 / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[abs(cat_neighbors['DEC'] - gal['DEC']) < 1.0 / dis / np.pi * 180]
        coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
        cat_neighbors = cat_neighbors[coord_neighbors.separation(coord_gal).degree < 1.0 / dis / np.pi * 180]

        if len(cat_neighbors) == 0:  # exclude central gals which has no companion
            massive_counts -= 1
            continue
        mass_neighbors = cat_neighbors['MASS_BEST']
        # if gal['MASS_BEST'] < max(cat_neighbors['MASS_BEST']):  # exclude central gals which has larger mass companion
        #     massive_counts -= 1
        #     continue

        mass_high = 12.
        mass_low = 9.5
        mass_neighbors_bkg = bkg(gal['RA'], np.copy(cat_neighbors_z_slice), coord_massive_gal)
        SMF_z += np.histogram(mass_neighbors, bins=bin_counts, range=(mass_low, mass_high))[0]
        SMF_z_bkg += np.histogram(mass_neighbors_bkg, bins=bin_counts, range=(mass_low, mass_high))[0]
        print(gal['ID'])
    print(massive_counts)

    ##################################  PLOT STELLAR MASS FUNCTIONS
    plt.figure(figsize=(7, 6))
    plt.rc('font', family='serif'), plt.rc('xtick', labelsize=15), plt.rc('ytick', labelsize=15)
    plt.plot(np.arange(mass_low, mass_high, (mass_high - mass_low)/bin_counts), SMF_z, '-o', alpha=0.4)
    plt.plot(np.arange(mass_low, mass_high, (mass_high - mass_low)/bin_counts), SMF_z_bkg, '-o', alpha=0.4)
    plt.plot(np.arange(mass_low, mass_high, (mass_high - mass_low)/bin_counts), SMF_z - SMF_z_bkg, '-o')
    plt.yscale('log')
    plt.xlabel('mass', fontsize=15)
    plt.ylabel('number counts per bin', fontsize=15)
    plt.savefig('SMFs/smf_'+str(z)+'_test.png')
    plt.close()