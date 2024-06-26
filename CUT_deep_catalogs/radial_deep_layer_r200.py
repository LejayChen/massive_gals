import sys
from random import random
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.cosmology import WMAP9
from astropy.table import Table
from astropy.io import fits
from scipy import stats
from sklearn.neighbors import KDTree
from mpi4py import MPI

comm = MPI.COMM_WORLD
n_procs = comm.Get_size()
rank = comm.Get_rank()


def bkg(cat_neighbors_z_slice_rand, coord_massive_gal_rand, mode='count'):
    # __init__
    global R_m_stack_bkg, R_u_stack_bkg, z
    counts_gals_rand = 0
    n = 0
    num_p = 1  # number of blank pointing's per central
    coord_rand_list = []
    num_before_success = 0
    flag_bkg = 0
    while n < num_p:  # get several blank pointing's to estimate background
        id_rand = int(random() * len(cat_random_copy))
        ra_rand = cat_random_copy[id_rand]['RA']
        dec_rand = cat_random_copy[id_rand]['DEC']
        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(ra_rand, dec_rand, unit="deg"), coord_massive_gal_rand, nthneighbor=1)

        num_before_success += 1
        if num_before_success > 100:
            flag_bkg = 1
            break

        if sep2d.degree > 1.4 / dis / np.pi * 180:  # make sure the random pointing is away from any central galaxy (blank)
            num_before_success = 0
            coord_rand = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)
            coord_rand_list.append(coord_rand)
            cat_neighbors_rand = cat_neighbors_z_slice_rand[abs(cat_neighbors_z_slice_rand['RA'] - ra_rand) < (r200/1000) / dis / np.pi * 180]
            cat_neighbors_rand = cat_neighbors_rand[abs(cat_neighbors_rand['DEC'] - dec_rand) < (r200/1000) / dis / np.pi * 180]
            coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)
            cat_neighbors_rand = cat_neighbors_rand[coord_neighbors_rand.separation(coord_rand).degree < (r200/1000) / dis / np.pi * 180]

            # make some cuts
            cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand[mass_keyname] > masscut_low]
            cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand[mass_keyname] < masscut_high]

            coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)
            radius_neighbors_rand = coord_neighbors_rand.separation(coord_rand).degree / 180. * np.pi * dis * 1000
            if len(radius_neighbors_rand) != 0:
                mass_neighbors_rand = cat_neighbors_rand[mass_keyname]
                if ssfq == 'all':
                    sfq_weights = np.ones(len(radius_neighbors_rand))
                elif ssfq == 'ssf':
                    sfq_weights = cat_neighbors_rand['sfProb']
                else:
                    sfq_weights = 1 - cat_neighbors_rand['sfProb']

                if mode == 'count':
                    counts_gals_rand += np.histogram(radius_neighbors_rand, weights=np.array(sfq_weights), bins=bin_edges*r200)[0]# smoothed background, assuming bkg is uniform
                    R_m, R_u = correct_for_masked_area(ra_rand, dec_rand)
                    R_m_stack_bkg += R_m
                    R_u_stack_bkg += R_u
                elif mode == 'mass':
                    binned_data_rand = stats.binned_statistic(radius_neighbors_rand, 10 ** (mass_neighbors_rand - 10), statistic='sum', bins=bin_edges*r200)
                    mass_binned_rand = binned_data_rand[0]
                    counts_gals_rand += mass_binned_rand
            else:
                counts_gals_rand += np.zeros(bin_number)
            n = n + 1

    return coord_rand_list, counts_gals_rand / float(num_p), flag_bkg


def cut_random_cat(cat_rand, coord_list):
    # cut random point catalog to avoid overlapping
    for coord in coord_list:
        coord_rand_list = SkyCoord(cat_rand['RA'] * u.deg, cat_rand['DEC'] * u.deg)
        cat_rand = cat_rand[coord_rand_list.separation(coord).degree > 1.4 / dis / np.pi * 180]
    add_to_random_points(coord_list)
    return cat_rand


def add_to_random_points(coord_list):
    # store coord of selected random points just for record
    for coord in coord_list:
        cat_random_points.add_row([coord.ra.value, coord.dec.value, gal['NUMBER']])
    return 0


def correct_for_masked_area(ra, dec):
    # correct area for normalization if it is partially in masked region

    cat_nomask = cat_random_nomask[abs(cat_random_nomask['RA'] - ra) < (r200/1000) / dis / np.pi * 180]
    cat_nomask = cat_nomask[abs(cat_nomask['DEC'] - dec) < (r200/1000) / dis / np.pi * 180]
    cat_nomask = cat_nomask[SkyCoord(cat_nomask['RA'] * u.deg, cat_nomask['DEC'] * u.deg).separation
                            (SkyCoord(ra * u.deg, dec * u.deg)).degree < (r200/1000) / dis / np.pi * 180]

    cat_mask = cat_random[abs(cat_random['RA'] - ra) < (r200/1000) / dis / np.pi * 180]
    cat_mask = cat_mask[abs(cat_mask['DEC'] - dec) < (r200/1000) / dis / np.pi * 180]
    cat_mask = cat_mask[SkyCoord(cat_mask['RA'] * u.deg, cat_mask['DEC'] * u.deg).separation
                        (SkyCoord(ra * u.deg, dec * u.deg)).degree < (r200/1000) / dis / np.pi * 180]

    if len(cat_nomask) == 0:
        return np.zeros(bin_number), np.zeros(bin_number)
    else:
        coord = SkyCoord(ra * u.deg, dec * u.deg)
        coord_nomask = SkyCoord(cat_nomask['RA'] * u.deg, cat_nomask['DEC'] * u.deg)
        radius_list_nomask = coord_nomask.separation(coord).degree / 180. * np.pi * dis * 1000
        count_nomask = np.histogram(radius_list_nomask, bins=bin_edges*r200)[0]
        count_nomask = np.array(count_nomask).astype(float)
        if len(cat_mask) == 0:
            count_mask = np.zeros(bin_number)
        else:
            coord_mask = SkyCoord(cat_mask['RA'] * u.deg, cat_mask['DEC'] * u.deg)
            radius_list_mask = coord_mask.separation(coord).degree / 180. * np.pi * dis * 1000
            count_mask = np.histogram(radius_list_mask, bins=bin_edges*r200)[0]
            count_mask = np.array(count_mask).astype(float)

        return count_mask, count_nomask
        # return np.ones(bin_number), np.ones(bin_number)


def spatial_comp(z, masscut_low, masscut_high, ssfq):
    path_curve = '/Users/lejay/research/massive_gals/completeness_curve/curves_graham/'
    try:
        comp_bootstrap = np.genfromtxt(
            path_curve + 'comp_bootstrap_all_' + ssfq + '_' + str(masscut_low) + '_' + str(masscut_high) + '_'
            + str(round(z - 0.1, 1)) + '_' + str(round(z + 0.1, 1)) + '.txt')
    except IOError:
        comp_bootstrap = np.genfromtxt(
            path_curve + 'comp_bootstrap_all_' + ssfq + '_' + str(round(z - 0.1, 1)) + '_'
            + str(round(z + 0.1, 1)) + '_' + str(masscut_low) + '_' + str(masscut_high) + '.txt')

    spatial_weight = 1. / np.nanmedian(comp_bootstrap, axis=0)
    spatial_weight_err = np.nanstd(comp_bootstrap, axis=0)/np.nanmedian(comp_bootstrap, axis=0)**2
    # return np.ones(bin_number), np.zeros(bin_number)
    return spatial_weight, spatial_weight_err

    
def cal_error(n_sample, n_bkg, n_central, w_comp, w_comp_err):
    n_comp = np.array(n_sample - n_bkg).astype(float)
    n_comp[n_comp==0] = 1

    if len(w_comp) != bin_number:
        w_comp = np.ones(bin_number)
        w_comp_err = np.zeros(bin_number)

    sigma_n_sample_ori = np.sqrt(n_sample/w_comp)
    sigma_n_sample_corr = np.sqrt(sigma_n_sample_ori**2*w_comp**2 + (n_sample/w_comp)**2*w_comp_err**2)
    sigma_n_bkg = np.sqrt(n_bkg)
    sigma_n_comp = np.sqrt(sigma_n_sample_corr**2+sigma_n_bkg**2)
    sigma_n_central = np.array(np.sqrt(n_central)).astype(float)

    errors = (n_comp/n_central)*np.sqrt((sigma_n_comp/n_comp)**2+(sigma_n_central/n_central)**2)
    return errors


def Ms_to_r200(log_Ms):
    M1 = 10 ** 12.52
    Ms = 10**log_Ms
    Ms0 = 10 ** 10.916
    beta = 0.457
    delta = 0.566
    gamma = 1.53
    rho_bar = 9.9e-30  # g/cm^3

    log_mh = np.log10(M1) + beta * np.log10(Ms / Ms0) + (Ms / Ms0) ** delta / (1 + (Ms / Ms0) ** (-1 * gamma)) - 0.5
    r200 = ((3*10**log_mh*1.989e30*1e3)/(800*np.pi*rho_bar))**(1/3)/3.086e21

    return r200  # in kpc


# ################# START #####################
cat_name = sys.argv[1]  # COSMOS_deep COSMOS_uddd ELAIS_deep XMM-LSS_deep DEEP_deep SXDS_uddd
mode = sys.argv[2]  # 'count' or 'mass'
z_keyname = 'zKDEPeak'
mass_keyname = 'MASS_MED'
masscut_low = 9.5
masscut_high = 13.0
path = 'total_sample_r200/'
prefix = ''
csfq = 'all'  # csf, cq, all
ssfq = sys.argv[3]  # ssf, sq, all
ssfr_cut = -10.5
sfprob_cut = 0.5
bin_number = 14

# read in data catalog
print('start reading catalogs', end='\r ')
cat = Table(fits.getdata('s16a_' + cat_name + '_masterCat.fits'))
cat = cat[cat[z_keyname] < 1]
cat_gal = cat[np.logical_and(cat['preds_median'] < 0.89, cat['inside'] == True)]
cat_gal = cat_gal[cat_gal[mass_keyname] > 8.0]

# read-in random point catalog
cat_random = Table.read('s16a_' + cat_name + '_random.fits')
try:
    cat_random = cat_random[cat_random['inside_u'] == True]
except KeyError:
    cat_random = cat_random[cat_random['inside_uS'] == True]
cat_random_nomask = np.copy(cat_random)
cat_random = cat_random[cat_random['MASK'] == False]

# to store position of selected random apertures
cat_random_points = Table(names=('RA', 'DEC', 'GAL_ID'))

# main loop
for z in [0.6]:
    z = round(z, 1)
    z_bin_size = 0.1
    masscut_low_host = 11.15
    masscut_high_host = 13.0
    print(mode, csfq, ssfq, masscut_low, masscut_high, masscut_low_host)
    print('=============' + str(round(z, 1)) + '================')
    cat_random_copy = np.copy(cat_random)  # reset random points catalog at each redshift

    cat_massive_gal = cat_gal[np.logical_and(cat_gal[mass_keyname] > masscut_low_host, cat_gal[mass_keyname] < masscut_high_host)]
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal[z_keyname] - z) < z_bin_size]
    coord_massive_gal = SkyCoord(cat_massive_z_slice['RA'] * u.deg, cat_massive_z_slice['DEC'] * u.deg)
    cat_all_z_slice = cat_gal[abs(cat_gal[z_keyname] - z) < 0.5]

    # initiations
    radial_dist = 0
    radial_dist_bkg = 0
    radial_count_dist = 0
    isolated_counts = len(cat_massive_z_slice)
    massive_count = 0
    bkg_count = 0
    bin_centers_stack = 0
    companion_count_stack = 0

    R_m_stack = np.ones(bin_number) * 1e-12  # number of random points (masked) in each bin
    R_u_stack = np.ones(bin_number) * 1e-12  # number of random points (unmasked) in each bin
    R_m_stack_bkg = np.ones(bin_number) * 1e-12  # number of random points (masked, bkg apertures) in each bin
    R_u_stack_bkg = np.ones(bin_number) * 1e-12  # number of random points (unmasked, bkg apertures) in each bin

    for gal in cat_massive_z_slice:
        # setting binning scheme
        r200 = Ms_to_r200(gal['MASS_MED'])
        if r200 > 2000:  # central gals which has no companion
            isolated_counts -= 1
            continue

        bin_edges = 10 ** np.linspace(-1.9, 0, num=bin_number + 1)
        areas = np.array([])
        for i in range(len(bin_edges[:-1])):
            areas = np.append(areas, (bin_edges[i + 1] ** 2 - bin_edges[i] ** 2) * np.pi)

        massive_count += 1
        print('Progress:' + str(massive_count) + '/' + str(len(cat_massive_z_slice)), end='\r')
        dis = WMAP9.angular_diameter_distance(gal[z_keyname]).value
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)

        # prepare neighbors catalog
        cat_neighbors_z_slice = cat_all_z_slice[abs(cat_all_z_slice[z_keyname] - gal[z_keyname]) < 1.5 * 0.044 * (1 + z)]
        cat_neighbors = cat_neighbors_z_slice[abs(cat_neighbors_z_slice['RA'] - gal['RA']) < (r200/1000) / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[abs(cat_neighbors['DEC'] - gal['DEC']) < (r200/1000) / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[cat_neighbors['NUMBER'] != gal['NUMBER']]

        ind = KDTree(np.array(cat_neighbors['RA', 'DEC']).tolist()).query_radius([(gal['RA'], gal['DEC'])], (r200/1000) / dis / np.pi * 180)
        cat_neighbors = cat_neighbors[ind[0]]

        if len(cat_neighbors) == 0:  # central gals which has no companion
            print(gal['NUMBER'])
            continue
        else:
            ind = KDTree(np.array(cat_neighbors['RA', 'DEC']).tolist()).query_radius([(gal['RA'], gal['DEC'])], 0.7 / dis / np.pi * 180)
            cat_neighbors_test_isolation = cat_neighbors[ind[0]]
            if len(cat_neighbors_test_isolation) == 0:  # central gals which has no companion
                print(gal['NUMBER'])
                continue

        # isolation cut on central
        if gal[mass_keyname] < max(cat_neighbors_test_isolation[mass_keyname]):  # no more-massive companions
            isolated_counts -= 1
            continue

        # cut on central SF/Q
        if csfq == 'csf' and gal['sfProb'] < sfprob_cut:
            isolated_counts -= 1
            continue
        elif csfq == 'cq' and gal['sfProb'] > sfprob_cut:
            isolated_counts -= 1
            continue

        # cut on companion sample (cut the final sample)
        cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] > masscut_low]
        cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] < masscut_high]
        mass_neighbors = cat_neighbors[mass_keyname]
        ssfr_neighbors = cat_neighbors['SSFR_BEST']
        if len(cat_neighbors) == 0:  # central gals which has no companion
            continue

        coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
        radius_list = coord_neighbors.separation(coord_gal).degree / 180. * np.pi * dis * 1000  # kpc

        # get mean radius value of companions in each bin and stack
        bin_stats = stats.binned_statistic(radius_list, radius_list, statistic='mean', bins=bin_edges)
        bin_centers = bin_stats[0]
        bin_centers = np.nan_to_num(bin_centers)
        for i in range(len(bin_centers)):
            if bin_centers[i] == 0:
                bin_centers[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
        bin_centers_stack += bin_centers * len(radius_list)
        companion_count_stack += len(radius_list)

        # Core Function: statistics #
        if ssfq == 'all':
            sfq_weights = np.ones(len(radius_list))
        elif ssfq == 'ssf':
            sfq_weights = cat_neighbors['sfProb']
        else:
            sfq_weights = 1 - cat_neighbors['sfProb']

        count_binned = np.histogram(radius_list, weights=np.array(sfq_weights), bins=bin_edges*r200)[0]
        sat_counts = np.array(count_binned, dtype='f8')

        R_m, R_u = correct_for_masked_area(gal['RA'], gal['DEC'])
        R_m_stack += R_m
        R_u_stack += R_u
        if mode == 'count':  # radial distribution histogram
            coord_random_list, sat_counts_bkg, flag_bkg = bkg(cat_neighbors_z_slice, coord_massive_gal, mode=mode)
            radial_dist += sat_counts
            if flag_bkg == 0:
                radial_dist_bkg += sat_counts_bkg
                bkg_count = bkg_count + 1
        else:                # mass distribution histogram
            binned_data = stats.binned_statistic(radius_list, 10 ** (mass_neighbors - 10), statistic='sum', bins=bin_edges*r200)
            mass_binned = binned_data[0]
            sat_masses = np.array(mass_binned, dtype='f8')
            coord_random_list, sat_masses_bkg, flag_bkg = bkg(cat_neighbors_z_slice, coord_massive_gal, mode=mode)
            radial_dist += sat_masses
            radial_count_dist += sat_counts
            if flag_bkg == 0:
                radial_dist_bkg += sat_masses_bkg
                bkg_count = bkg_count + 1
        cat_random_copy = cut_random_cat(cat_random_copy, coord_random_list)

    # detection completeness correction
    # spatial_weight, spatial_weight_err = spatial_comp(z, masscut_low, masscut_high, ssfq)
    spatial_weight, spatial_weight_err = np.ones(bin_number), np.zeros(bin_number)
    # radial_dist = radial_dist * spatial_weight
    # radial_dist_bkg = radial_dist_bkg * np.average(spatial_weight[-3:])

    # aggregate results
    radial_dist_norm = radial_dist * R_u_stack / R_m_stack / float(isolated_counts) / areas
    radial_dist_bkg_norm = radial_dist_bkg * R_u_stack_bkg / R_m_stack_bkg / float(bkg_count) /areas

    if mode == 'count':
        err = cal_error(radial_dist, radial_dist_bkg, isolated_counts, spatial_weight, spatial_weight_err) / areas
    else:
        err = cal_error(radial_dist, radial_dist_bkg, isolated_counts, spatial_weight, spatial_weight_err) \
              * (radial_dist / radial_count_dist) / areas
    n_central, n_count, n_err = isolated_counts, radial_dist_norm - radial_dist_bkg_norm, err
    result = np.append([n_central], [n_count, n_err])

    # output result to file
    print('Finished gals: '+str(isolated_counts)+'/'+str(len(cat_massive_z_slice)))
    print('Finished bkgs: '+str(bkg_count))
    filename = path + prefix + str(mode) + cat_name + '_' + str(masscut_low) + '_' + str(csfq) + '_' \
               + str(ssfq) + '_' + str(round(z, 1)) + '.txt'
    np.save(path + prefix + 'bin_edges', bin_edges)
    np.save(path + prefix + 'bin_centers', bin_centers_stack / companion_count_stack)
    if sum(R_u_stack) < 2e-12:
        np.savetxt(filename, result)
        print('No massive galaxy with desired satellites!')
    else:
        np.savetxt(filename, result)
