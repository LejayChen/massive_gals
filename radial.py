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

    # __init__
    counts_gals_rand = np.zeros(19)
    n = 0
    num_p = 3  # number of blank pointing's
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
        if sep2d.degree > 1.0/dis/np.pi*180:  # make sure the random pointing is away from any central galaxy (blank)
            # cut all objects catalog to get neighbor catalog
            coord_all_z_slice_rand = SkyCoord(cat_all_z_slice_rand['RA']*u.deg, cat_all_z_slice_rand['DEC']*u.deg)
            coord_rand = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)
            cat_neighbors_rand = cat_all_z_slice_rand[coord_all_z_slice_rand.separation(coord_rand).degree < 0.7/dis/np.pi*180]
            if len(cat_neighbors_rand) == 0:
                continue  # no gals (in masked region)

            coord_rand = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)
            radius_neighbors_rand = coord_neighbors.separation(coord_rand).degree / 180. * np.pi * dis * 1000
            count_gal_rand, edges_rand = np.histogram(radius_neighbors_rand, bins=10**np.linspace(1, 2.75, num=20))
            counts_gals_rand += count_gal_rand
            n = n + 1

    return counts_gals_rand/float(num_p)

for z in np.arange(7, 7.1)/10.:
    print('============='+str(z)+'================')
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal['ZPHOT'] - z) < 0.3]
    cat_all_z_slice = cat_gal[abs(cat_gal['ZPHOT'] - z) < 0.6]

    # Fetch coordinates for massive gals
    cat_massive_z_slice['RA'].unit = u.deg
    cat_massive_z_slice['DEC'].unit = u.deg
    coord_massive_gal = SkyCoord.guess_from_table(cat_massive_z_slice)

    radial_dist = 0
    radial_dist_err = 0
    radius_max = 600
    massive_counts = len(cat_massive_z_slice)
    for gal in cat_massive_z_slice:
        print(gal['ID'])
        dis = WMAP9.angular_diameter_distance(gal['ZPHOT']).value
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)
        cat_neighbors_z_slice = cat_all_z_slice[abs(cat_all_z_slice['ZPHOT'] - gal['ZPHOT']) < 0.6]
        coord_neighbors_z_slice = SkyCoord(cat_neighbors_z_slice['RA'] * u.deg, cat_neighbors_z_slice['DEC'] * u.deg)
        cat_neighbors = cat_neighbors_z_slice[coord_neighbors_z_slice.separation(coord_gal).degree < 0.7 / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[cat_neighbors['ID'] != gal['ID']]

        if len(cat_neighbors) == 0:  # exclude central gals which has no companion
            massive_counts -= 1
            continue
        mass_neighbors = cat_neighbors['MASS_BEST']
        if gal['MASS_BEST'] < max(cat_neighbors['MASS_BEST']):  # no more-massive companions
            massive_counts -= 1
            continue

        coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
        radius_list = coord_neighbors.separation(coord_gal).degree / 180. * np.pi * dis * 1000  # kpc
        count_binned, bin_edges = np.histogram(radius_list, bins=10**np.linspace(1, 2.75, num=20))
        bin_left_edges = bin_edges[:-1]

        sat_counts = np.array(count_binned, dtype='f8') - bkg(gal['RA'], cat_all_z_slice, coord_massive_gal)  # counts
        sat_counts_err = np.sqrt(sat_counts)  # noise (assume poisson)

        areas = np.array([])
        for i in range(len(bin_edges[:-1])):
            areas = np.append(areas, (bin_edges[i + 1] ** 2 - bin_edges[i] ** 2) * np.pi)
        radial_dist += sat_counts / areas
        radial_dist_err += sat_counts_err / areas

    plt.plot(bin_left_edges, radial_dist/float(massive_counts))
    plt.annotate(str(massive_counts), (1, 1), color='red', fontsize=14, xycoords='axes points')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('radial_test.png')
    plt.show()