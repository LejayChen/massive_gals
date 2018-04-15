from astropy.table import Table
from astropy.cosmology import WMAP9
from astropy.coordinates import SkyCoord, match_coordinates_sky
from random import random
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

cat = Table.read('CUT2_CLAUDS_HSC_VISTA_Ks23.3_PHYSPARAM_TM.fits')
cat_gal = cat[cat['CLASS'] == 0]
cat_massive_gal = cat_gal[cat_gal['MASS_BEST'] > 11.0]

def bkg(ra_central, cat_all_z_slice_rand, coord_massive_gal_rand):
    '''
    return a background correction for each central galaxy by looking at blank pointings without massive galaxies.
    mass_central: mass of the central galaxy,  ra_central: ra value for central galaxy. if ra_central>100, gal in COSMOS, if ra_central<100, gal in XMM-LSS
    '''
    mass_sat_rand = []
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

        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(ra_rand, dec_rand, unit="deg"), coord_massive_gal_rand, nthneighbor=1)
        if sep2d.degree > 1.5/dis/np.pi*180:  # make sure the random pointing is away from any central galaxy (blank)
            # cut all objects catalog to get neighbor catalog
            coord_all_z_slice_rand = SkyCoord(cat_all_z_slice_rand['RA']*u.deg, cat_all_z_slice_rand['DEC']*u.deg)
            coord_rand = SkyCoord(ra_rand*u.deg, dec_rand*u.deg)
            cat_neighbors_rand = cat_all_z_slice_rand[coord_all_z_slice_rand.separation(coord_rand).degree < 0.6/dis/np.pi*180]
            if len(cat_neighbors_rand) == 0:
                continue  # no gals (in masked region)
            # retrieve mass info in sat catalog (all, sf & q)
            # calculate total mass for satellites in blank pointing's (all, sf & q)
            mass_neighbors_rand = cat_neighbors_rand['MASS_BEST']
            mass_sat_rand.append(np.sum(10 ** (mass_neighbors_rand[mass_neighbors_rand > 10] - 8)))  # unit 10**8 M_sun
            n = n + 1

    return mass_neighbors_rand

bin_counts = 20
for z in np.arange(7, 7.1, 1)/10.:
    print('============='+str(z)+'================')
    dis = WMAP9.angular_diameter_distance(z).value  # angular diameter distance at redshift z
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal['ZPHOT'] - z) < 0.2]  # massive galaxies in this z slice
    cat_all_z_slice = cat_gal[abs(cat_gal['ZPHOT'] - z) < 0.2]  # all galaxies in this z slice (large than delta z, will be cut further)

    # Fetch coordinates for massive gals
    cat_massive_z_slice['RA'].unit = u.deg
    cat_massive_z_slice['DEC'].unit = u.deg
    coord_massive_gal = SkyCoord.guess_from_table(cat_massive_z_slice)

    massive_counts = len(cat_massive_z_slice)
    SMF_z = np.zeros(bin_counts)
    SMF_z_bkg = np.zeros(bin_counts)
    for gal in cat_massive_z_slice:
        print(gal['ID'])
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)

        cat_neighbors_z_slice = np.copy(cat_all_z_slice)
        coord_neighbors_z_slice = SkyCoord(cat_neighbors_z_slice['RA'] * u.deg, cat_neighbors_z_slice['DEC'] * u.deg)
        cat_neighbors = cat_neighbors_z_slice[coord_neighbors_z_slice.separation(coord_gal).degree < 0.6 / dis / np.pi * 180]

        if len(cat_neighbors) == 0:  # exclude central gals which has no companion
            massive_counts -= 1
            continue

        # retrieve mass info in sat catalog (all, sf & q)
        mass_neighbors = cat_neighbors['MASS_BEST']
        if gal['MASS_BEST'] < max(cat_neighbors['MASS_BEST']):  # exclude central gals which has larger mass companion
            massive_counts -= 1
            continue

        mass_neighbors_bkg = bkg(gal['RA'], np.copy(cat_neighbors_z_slice), coord_massive_gal)

        mass_high = 12.
        mass_low = 9.5
        SMF_z += np.histogram(mass_neighbors, bins=bin_counts, range=(mass_low, mass_high))[0]
        SMF_z_bkg += np.histogram(mass_neighbors_bkg, bins=bin_counts, range=(mass_low, mass_high))[0]

    ##################################  PLOT STELLAR MASS FUNCTIONS
    plt.figure(figsize=(7, 6))
    plt.rc('font', family='serif'), plt.rc('xtick', labelsize=15), plt.rc('ytick', labelsize=15)
    plt.plot(np.arange(mass_low, mass_high, (mass_high - mass_low)/bin_counts), SMF_z, '-o', alpha=0.4)
    plt.plot(np.arange(mass_low, mass_high, (mass_high - mass_low)/bin_counts), SMF_z_bkg, '-o', alpha=0.4)
    plt.plot(np.arange(mass_low, mass_high, (mass_high - mass_low)/bin_counts), SMF_z - SMF_z_bkg, '-o')
    plt.xlabel('mass', fontsize=14)
    plt.ylabel('number density', fontsize=14)
    plt.savefig('SMFs/smf_'+str(z)+'_test.png')
    plt.close()
