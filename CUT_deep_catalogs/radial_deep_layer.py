from astropy.table import Table
from astropy.cosmology import WMAP9
from astropy.coordinates import SkyCoord, match_coordinates_sky
from sklearn.neighbors import KDTree
from random import random
from scipy import stats
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import sys


def bkg(cat_neighbors_z_slice_rand, coord_massive_gal_rand, mode='count'):
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
        id_rand = int(random() * len(cat_random))
        ra_rand = cat_random[id_rand]['RA']
        dec_rand = cat_random[id_rand]['DEC']
        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(ra_rand, dec_rand, unit="deg"), coord_massive_gal_rand, nthneighbor=1)
        if sep2d.degree > 1.4/dis/np.pi*180:  # make sure the random pointing is away from any central galaxy (blank)
            coord_rand = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)
            coord_rand_list.append(coord_rand)
            cat_neighbors_rand = cat_neighbors_z_slice_rand[abs(cat_neighbors_z_slice_rand['RA'] - ra_rand) < 0.7/dis/np.pi*180]
            cat_neighbors_rand = cat_neighbors_rand[abs(cat_neighbors_rand['DEC'] - dec_rand) < 0.7/dis/np.pi*180]
            coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)
            cat_neighbors_rand = cat_neighbors_rand[coord_neighbors_rand.separation(coord_rand).degree < 0.7/dis/np.pi*180]

            # cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['MASS_MED']>10.2]
            cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['SSFR_BEST']>-11]

            coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)
            radius_neighbors_rand = coord_neighbors_rand.separation(coord_rand).degree/180.*np.pi*dis*1000
            if len(radius_neighbors_rand) != 0:
                if mode == 'count':
                    count_gal_rand, edges_rand = np.histogram(radius_neighbors_rand, bins=radial_bins)
                    counts_gals_rand += count_gal_rand/correct_for_masked_area(ra_rand, dec_rand)
                elif mode == 'mass':
                    mass_neighbors_rand = cat_neighbors_rand['MASS_MED']
                    binned_data_rand = stats.binned_statistic(radius_neighbors_rand, 10**(mass_neighbors_rand - 10), statistic='sum', bins=radial_bins)
                    mass_binned_rand = binned_data_rand[0]
                    counts_gals_rand += mass_binned_rand
            else:
                counts_gals_rand += np.zeros(14)
            n = n + 1

    return coord_rand_list, counts_gals_rand/float(num_p)


def cut_random_cat(cat_rand, coord_list):
    for coord in coord_list:
        coord_rand_list = SkyCoord(cat_rand['RA'] * u.deg, cat_rand['DEC'] * u.deg)
        cat_rand = cat_rand[coord_rand_list.separation(coord).degree > 1.2 / dis / np.pi * 180]
    return cat_rand


def add_to_random_points(coord_list):
    for coord in coord_list:
        cat_random_points.add_row([coord.ra.value, coord.dec.value, gal['NUMBER']])
    return 0


def correct_for_masked_area(ra,dec):
    cat_nomask = cat_random_nomask[abs(cat_random_nomask['RA']-ra) < 0.7 / dis / np.pi * 180]
    cat_nomask = cat_nomask[abs(cat_nomask['DEC'] - dec) < 0.7 / dis / np.pi * 180]
    cat_nomask = cat_nomask[SkyCoord(cat_nomask['RA']*u.deg, cat_nomask['DEC']*u.deg).separation
                            (SkyCoord(ra*u.deg, dec*u.deg)).degree < 0.7/dis/np.pi*180]

    cat_mask = cat_random[abs(cat_random['RA']-ra) < 0.7 / dis / np.pi * 180]
    cat_mask = cat_mask[abs(cat_mask['DEC'] - dec) < 0.7 / dis / np.pi * 180]
    cat_mask = cat_mask[SkyCoord(cat_mask['RA'] * u.deg, cat_mask['DEC'] * u.deg).separation
                            (SkyCoord(ra * u.deg, dec * u.deg)).degree < 0.7 / dis / np.pi * 180]

    count_nomask = np.histogram(cat_nomask, bins=radial_bins)[0]
    count_mask = np.histogram(cat_mask, bins=radial_bins)[0]

    area_correction = count_mask/count_nomask
    return area_correction


# ################# START #####################
print('start reading catalogs', end='\r')
cat_name = sys.argv[1]  # COSMOS_deep COSMOS_uddd ELAIS_deep XMM-LSS_deep DEEP_deep SXDS_uddd
mode = sys.argv[2]  # 'count' or 'mass'
cat = Table.read('CUT_'+cat_name+'.fits')
cat = cat[cat['zKDEPeak'] < 1]
cat_gal = cat[cat['preds_median'] < 0.89]
cat_gal = cat_gal[cat_gal['MASS_MED'] > 9.0]
cat_massive_gal = cat_gal[cat_gal['MASS_MED'] > 11.15]
cat_random = Table.read('s16a_' + cat_name + '_random.fits')
cat_random_nomask = np.copy(cat_random)
cat_random = cat_random[cat_random['MASK'] == False]
try:
    cat_random = cat_random[cat_random['inside_u'] == True]
except KeyError:
    cat_random = cat_random[cat_random['inside_uS'] == True]
cat_random_points = Table(names=('RA', 'DEC', 'GAL_ID'))

radial_bins = 10 ** np.linspace(1, 2.75, num=15)
for z in np.arange(3, 6.1, 3)/10.:
    print('============='+str(z)+'================')
    # set up random points catalog
    cat_random_copy = np.copy(cat_random)

    # slice random points catalog
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal['zKDEPeak'] - z) < 0.09]
    # cat_massive_z_slice = cat_massive_z_slice[np.random.rand(len(cat_massive_z_slice)) < 0.9]
    cat_all_z_slice = cat_gal[abs(cat_gal['zKDEPeak'] - z) < 0.5]

    # Fetch coordinates for massive gals
    coord_massive_gal = SkyCoord(cat_massive_z_slice['RA']*u.deg, cat_massive_z_slice['DEC']*u.deg)

    radial_dist = 0
    radial_dist_bkg = 0
    massive_counts = len(cat_massive_z_slice)
    massive_count = 0
    bin_centers_stack = 0
    companion_count_stack = 0
    for gal in cat_massive_z_slice:
        massive_count += 1
        print('Progress:'+str(massive_count)+'/'+str(len(cat_massive_z_slice)), end='\r')
        dis = WMAP9.angular_diameter_distance(gal['zKDEPeak']).value
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)

        # prepare neighbors catalog
        cat_neighbors_z_slice = cat_all_z_slice[abs(cat_all_z_slice['zKDEPeak'] - gal['zKDEPeak']) < 1.5 * 0.03 * (1 + z)]
        cat_neighbors = cat_neighbors_z_slice[abs(cat_neighbors_z_slice['RA'] - gal['RA']) < 0.7 / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[abs(cat_neighbors['DEC'] - gal['DEC']) < 0.7 / dis / np.pi * 180]
        if len(cat_neighbors) == 0:  # exclude central gals which has no companion
            coord_random_list, radial_bkg = bkg(cat_neighbors_z_slice, coord_massive_gal, mode=mode)
            radial_dist_bkg += radial_bkg
            cat_random_copy = cut_random_cat(cat_random_copy, coord_random_list)
            add_to_random_points(coord_random_list)
            continue
        else:
            ind = KDTree(np.array(cat_neighbors['RA', 'DEC']).tolist()).query_radius([(gal['RA'], gal['DEC'])], 0.7 / dis / np.pi * 180)
            cat_neighbors = cat_neighbors[ind[0]]
            cat_neighbors = cat_neighbors[cat_neighbors['NUMBER'] != gal['NUMBER']]
            if len(cat_neighbors) == 0:  # exclude central gals which has no companion
                coord_random_list, radial_bkg = bkg(cat_neighbors_z_slice, coord_massive_gal, mode=mode)
                radial_dist_bkg += radial_bkg
                cat_random_copy = cut_random_cat(cat_random_copy, coord_random_list)
                add_to_random_points(coord_random_list)
                continue

        mass_neighbors = cat_neighbors['MASS_MED']
        if gal['MASS_MED'] < max(cat_neighbors['MASS_MED']):  # no more-massive companions
            massive_counts -= 1
            continue

        if gal['SSFR_BEST'] > -11:
            massive_counts -= 1
            continue

        # ###### more cuts ######
        # cat_neighbors = cat_neighbors[cat_neighbors['MASS_MED']>10.2]
        cat_neighbors = cat_neighbors[cat_neighbors['SSFR_BEST']>-11]


        if len(cat_neighbors) == 0:  # exclude central gals which has no companion
            coord_random_list, radial_bkg = bkg(cat_neighbors_z_slice, coord_massive_gal, mode=mode)
            radial_dist_bkg += radial_bkg
            cat_random_copy = cut_random_cat(cat_random_copy, coord_random_list)
            add_to_random_points(coord_random_list)
            continue
        coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
        radius_list = coord_neighbors.separation(coord_gal).degree / 180. * np.pi * dis * 1000  # kpc

        # get mean radius value of companions in each bin and stack
        bin_stats = stats.binned_statistic(radius_list, radius_list, statistic='mean', bins=radial_bins)
        bin_centers = bin_stats[0]
        bin_edges = bin_stats[1]
        bin_centers = np.nan_to_num(bin_centers)
        for i in range(len(bin_centers)):
            if bin_centers[i] == 0:
                bin_centers[i] = (bin_edges[i]+bin_edges[i+1])/2
        bin_centers_stack += bin_centers*len(radius_list)
        companion_count_stack += len(radius_list)

        if mode == 'count':
            # radial distribution histogram
            count_binned = np.histogram(radius_list, bins=10**np.linspace(1, 2.75, num=15))[0]
            sat_counts = np.array(count_binned, dtype='f8')
            coord_random_list, sat_counts_bkg = bkg(cat_neighbors_z_slice, coord_massive_gal, mode=mode)
            cat_random_copy = cut_random_cat(cat_random_copy, coord_random_list)  # counts
            add_to_random_points(coord_random_list)
            radial_dist += sat_counts/correct_for_masked_area(gal['RA'], gal['DEC'])
            radial_dist_bkg += sat_counts_bkg
        else:
            # mass distribution histogram
            binned_data = stats.binned_statistic(radius_list, 10**(mass_neighbors-10), statistic='sum', bins=radial_bins)
            mass_binned = binned_data[0]
            sat_masses = np.array(mass_binned, dtype='f8')
            coord_random_list, sat_masses_bkg = bkg(cat_neighbors_z_slice, coord_massive_gal, mode=mode)
            cat_random_copy = cut_random_cat(cat_random_copy, coord_random_list)
            add_to_random_points(coord_random_list)
            radial_dist += sat_masses
            radial_dist_bkg += sat_masses_bkg

    areas = np.array([])
    for i in range(len(bin_edges[:-1])):
        areas = np.append(areas, (bin_edges[i + 1] ** 2 - bin_edges[i] ** 2) * np.pi)


    np.save('split_sfq_mass_data/'+str(mode)+cat_name+'_cq_'+'ssf_'+str(z), (radial_dist-radial_dist_bkg)/areas/float(massive_counts))
    np.save('split_sfq_mass_data/'+str(mode)+cat_name+'_cq_'+'ssf_'+str(z)+'_err', np.sqrt(radial_dist+radial_dist_bkg)/areas/float(massive_counts))
    np.save('bin_centers', bin_centers_stack/companion_count_stack)
    np.save('bin_edges', bin_edges)
    # cat_random_points.write('random_points_'+cat_name+'_nonoverlapping.fits', overwrite=True)
    print('massive counts:', len(cat_massive_z_slice), massive_counts)

    # fig = plt.figure(figsize=(9, 6))
    # plt.rc('font', family='serif'), plt.rc('xtick', labelsize=15), plt.rc('ytick', labelsize=15)
    # plt.errorbar(bin_edges[:-1], (radial_dist-radial_dist_bkg)/areas/float(massive_counts), fmt='.-k', yerr=np.sqrt(radial_dist+radial_dist_bkg)/areas/float(massive_counts))
    # plt.annotate(str(massive_counts), (1, 1), color='red', fontsize=14, xycoords='axes points')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Projected Radius [kpc]', fontsize=14)
    # if mode == 'mass':
    #     plt.ylabel(r'M/10$^{10}$M$_\odot$ kpc$^{-2}$ UMG$^{-1}$ dr', fontsize=14)
    #     plt.savefig('radial_mass_'+cat_name+'.png')
    # else:
    #     plt.ylabel(r'N kpc$^{-2}$ UMG$^{-1}$ dr', fontsize=14)
    #     plt.savefig('radial_count_'+cat_name+'.png')
    # plt.show()