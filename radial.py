from astropy.table import Table
from astropy.cosmology import WMAP9
from astropy.coordinates import SkyCoord, match_coordinates_sky
from sklearn.neighbors import KDTree
from random import random
from scipy import stats
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from radial_functions import *


def bkg(ra_central, cat_neighbors_z_slice_rand, coord_massive_gal_rand, mode='count'):
    '''
    return a background correction for each central galaxy by looking at blank pointings without massive galaxies.
    mass_central: mass of the central galaxy,  ra_central: ra value for central galaxy. if ra_central>100, gal in COSMOS, if ra_central<100, gal in XMM-LSS
    '''
    # __init__
    counts_gals_rand = 0
    n = 0
    num_p = 1  # number of blank pointing's
    coord_rand_list = []
    while n < num_p:  # get several blank pointing's to estimate background
        id_rand = int(random() * len(cat_random_copy))
        ra_rand = cat_random_copy[id_rand]['ra'] + random() * 1.5 / dis / np.pi * 180
        dec_rand = cat_random_copy[id_rand]['dec'] + random() * 1.5 / dis / np.pi * 180
        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(ra_rand, dec_rand, unit="deg"), coord_massive_gal_rand, nthneighbor=1)
        if sep2d.degree > 1.4/dis/np.pi*180:  # make sure the random pointing is away from any central galaxy (blank)
            coord_rand = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)
            cat_neighbors_rand = cat_neighbors_z_slice_rand[abs(cat_neighbors_z_slice_rand['RA'] - ra_rand) < 0.7/dis/np.pi*180]
            cat_neighbors_rand = cat_neighbors_rand[abs(cat_neighbors_rand['DEC'] - dec_rand) < 0.7/dis/np.pi*180]
            coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)
            cat_neighbors_rand = cat_neighbors_rand[coord_neighbors_rand.separation(coord_rand).degree < 0.7/dis/np.pi*180]
            if len(cat_neighbors_rand) == 0: continue  # no gals (in masked region)

            if ssfq == 'ssf':
                cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['SSFR_BEST'] > -11]
            elif ssfq == 'sq':
                cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['SSFR_BEST'] < -11]
            coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)
            radius_neighbors_rand = coord_neighbors_rand.separation(coord_rand).degree/180.*np.pi*dis*1000
            if mode == 'count':
                count_gal_rand, edges_rand = np.histogram(radius_neighbors_rand, bins=radial_bins)
                counts_gals_rand += count_gal_rand
            elif mode == 'mass':
                mass_neighbors_rand = cat_neighbors_rand['MASS_MED']
                binned_data_rand = stats.binned_statistic(radius_neighbors_rand, 10**(mass_neighbors_rand - 10), statistic='sum', bins=radial_bins)
                mass_binned_rand = binned_data_rand[0]
                counts_gals_rand += mass_binned_rand

            coord_rand_list.append(SkyCoord(ra_rand*u.deg, dec_rand*u.deg))
            n = n + 1

    return counts_gals_rand/float(num_p), coord_rand_list


def cut_random_cat(cat_rand, coord_list):
    # cut random point catalog to avoid overlapping
    for coord in coord_list:
        coord_rand_list = SkyCoord(cat_rand['ra'] * u.deg, cat_rand['dec'] * u.deg)
        cat_rand = cat_rand[coord_rand_list.separation(coord).degree > 1.2 / dis / np.pi * 180]
    add_to_random_points(coord_list)
    return cat_rand


def add_to_random_points(coord_list):
    # store coord of random points just for record
    for coord in coord_list:
        cat_random_points.add_row([coord.ra.value, coord.dec.value, gal['ID']])
    return 0

radial_bins = 10 ** np.linspace(1, 2.75, num=10)
masscut_low = 9.5
masscut_high = np.inf
ssfq = 'sq'
path = 'calibration_uddd/'
cat = Table.read('CUT3_CLAUDS_HSC_VISTA_Ks23.3_PHYSPARAM_TM.fits')
cat_random = Table.read('matched_RandomCat_uddd_IR.fits')
cat_gal = cat[cat['CLASS'] == 0]
cat_gal = cat_gal[cat_gal['MASS_MED'] > masscut_low]
cat_massive_gal = cat_gal[cat_gal['MASS_MED'] > 11.15]

mode = 'count'
cat_random_points = Table(names=('RA', 'DEC', 'GAL_ID'))
for z in np.arange(6, 6.1, 3)/10.:
    print('============='+str(z)+'================')
    cat_random_copy = np.copy(cat_random)
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal['ZPHOT'] - z) < 0.09]
    # cat_massive_z_slice = cat_massive_z_slice[np.random.rand(len(cat_massive_z_slice)) < 0.5]
    cat_all_z_slice = cat_gal[abs(cat_gal['ZPHOT'] - z) < 0.5]

    # Fetch coordinates for massive gals
    cat_massive_z_slice['RA'].unit = u.deg
    cat_massive_z_slice['DEC'].unit = u.deg
    coord_massive_gal = SkyCoord.guess_from_table(cat_massive_z_slice)

    radial_dist = 0
    radial_dist_bkg = 0
    massive_counts = len(cat_massive_z_slice)
    for gal in cat_massive_z_slice:
        dis = WMAP9.angular_diameter_distance(gal['ZPHOT']).value
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)

        # prepare neighbors catalog
        cat_neighbors_z_slice = cat_all_z_slice[abs(cat_all_z_slice['ZPHOT'] - gal['ZPHOT']) < 1.5 * 0.03 * (1 + z)]
        cat_neighbors = cat_neighbors_z_slice[abs(cat_neighbors_z_slice['RA'] - gal['RA']) < 0.7 / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[abs(cat_neighbors['DEC'] - gal['DEC']) < 0.7 / dis / np.pi * 180]
        if len(cat_neighbors) == 0:  # exclude central gals which has no companion
            radial_bkg, coord_bkg_list = bkg(gal['RA'], cat_neighbors_z_slice, coord_massive_gal, mode=mode)
            radial_dist_bkg += radial_bkg
            cat_random_copy = cut_random_cat(cat_random_copy, coord_bkg_list)
            continue

        else:
            ind = KDTree(np.array(cat_neighbors['RA', 'DEC']).tolist()).query_radius([(gal['RA'], gal['DEC'])], 0.7 / dis / np.pi * 180)
            cat_neighbors = cat_neighbors[ind[0]]
            cat_neighbors = cat_neighbors[cat_neighbors['ID'] != gal['ID']]
            if len(cat_neighbors) == 0:  # exclude central gals which has no companion
                radial_bkg, coord_bkg_list = bkg(gal['RA'], cat_neighbors_z_slice, coord_massive_gal, mode=mode)
                radial_dist_bkg += radial_bkg
                cat_random_copy = cut_random_cat(cat_random_copy, coord_bkg_list)
                continue

        mass_neighbors = cat_neighbors['MASS_MED']
        if gal['MASS_MED'] < max(cat_neighbors['MASS_MED']):  # no more-massive companions
            massive_counts -= 1
            continue

        print(gal['ID'])
        if ssfq == 'ssf':
            cat_neighbors = cat_neighbors[cat_neighbors['SSFR_BEST'] > -11]
        elif ssfq == 'sq':
            cat_neighbors = cat_neighbors[cat_neighbors['SSFR_BEST'] < -11]
        coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
        radius_list = coord_neighbors.separation(coord_gal).degree / 180. * np.pi * dis * 1000  # kpc
        if mode == 'count':
            # radial distribution histogram
            count_binned, bin_edges = np.histogram(radius_list, bins=radial_bins)
            sat_counts = np.array(count_binned, dtype='f8')
            sat_counts_bkg, coord_bkg_list = bkg(gal['RA'], cat_neighbors_z_slice, coord_massive_gal, mode=mode)  # counts
            radial_dist += sat_counts
            radial_dist_bkg += sat_counts_bkg
        else:
            # mass distribution histogram
            binned_data = stats.binned_statistic(radius_list, 10**(mass_neighbors-10), statistic='sum', bins=radial_bins)
            mass_binned = binned_data[0]
            bin_edges = binned_data[1]
            sat_masses = np.array(mass_binned, dtype='f8')
            sat_masses_bkg, coord_bkg_list = bkg(gal['RA'], cat_neighbors_z_slice, coord_massive_gal, mode=mode)
            radial_dist += sat_masses
            radial_dist_bkg += sat_masses_bkg

        cat_random_copy = cut_random_cat(cat_random_copy, coord_bkg_list)

    areas = np.array([])
    for i in range(len(bin_edges[:-1])):
        areas = np.append(areas, (bin_edges[i + 1] ** 2 - bin_edges[i] ** 2) * np.pi)

    filename = path + 'new_' + str(mode)+'_'+str(masscut_low)+'_'+str(z)
    np.save(filename, (radial_dist-radial_dist_bkg)/areas/float(massive_counts))
    np.save(filename+'_err', cal_error2(radial_dist, radial_dist_bkg, massive_counts))
    np.save(path + 'new_bin_edges', bin_edges)
    print('massive counts:', len(cat_massive_z_slice), massive_counts)


    # cat_random.write('random_points.fits', overwrite=True)
    # fig = plt.figure(figsize=(9, 6))
    # plt.rc('font', family='serif'), plt.rc('xtick', labelsize=15), plt.rc('ytick', labelsize=15)
    # plt.errorbar(bin_edges[:-1], (radial_dist-radial_dist_bkg)/areas/float(massive_counts), fmt='.-k', yerr=np.sqrt(radial_dist+radial_dist_bkg)/areas/float(massive_counts))
    # plt.annotate(str(massive_counts), (1, 1), color='red', fontsize=14, xycoords='axes points')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Projected Radius [kpc]', fontsize=14)
    # if mode == 'mass':
    #     plt.ylabel(r'M/10$^{10}$M$_\odot$ kpc$^{-2}$ UMG$^{-1}$ dr', fontsize=14)
    #     plt.savefig('radial_mass_test.png')
    # else:
    #     plt.ylabel(r'N kpc$^{-2}$ UMG$^{-1}$ dr', fontsize=14)
    #     plt.savefig('radial_count_test.png')
    # plt.show()