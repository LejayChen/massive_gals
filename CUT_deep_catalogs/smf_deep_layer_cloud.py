import sys
from random import random
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.cosmology import WMAP9
from astropy.stats import bootstrap
from astropy.table import Table
from sklearn.neighbors import KDTree
from mpi4py import MPI


def bkg(cat_neighbors_z_slice_rand, coord_massive_gal_rand, mass_cen):
    global z
    counts_gals_rand = np.zeros(bin_number)
    n = 0
    num_before_success = 0
    flag_bkg = 0

    coord_rand_list = []
    while n < 1:  # get several blank pointing's to estimate background
        id_rand = int(random() * len(cat_random))
        ra_rand = cat_random[id_rand]['RA']
        dec_rand = cat_random[id_rand]['DEC']
        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(ra_rand, dec_rand, unit="deg"), coord_massive_gal_rand, nthneighbor=1)

        num_before_success += 1
        if num_before_success > 20:
            flag_bkg = 1
            break

        if sep2d.degree > r_iso*2/dis/np.pi*180:  # make sure the random pointing is away from any central galaxy (blank)

            coord_rand = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)
            coord_rand_list.append(coord_rand)
            cat_neighbors_rand = cat_neighbors_z_slice_rand[abs(cat_neighbors_z_slice_rand['RA'] - ra_rand) < r_iso/dis/np.pi*180]
            cat_neighbors_rand = cat_neighbors_rand[abs(cat_neighbors_rand['DEC'] - dec_rand) < r_iso/dis/np.pi*180]
            coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)

            # exclude bkg apertures that contains galaxies more massive than central
            if len(cat_neighbors_rand) != 0:
                if max(cat_neighbors_rand['MASS_MED']) > mass_cen:
                    continue
                elif len(cat_neighbors_rand) < 100:
                    continue

            # choose radial range
            cat_neighbors_rand = cat_neighbors_rand[np.logical_and(coord_neighbors_rand.separation(coord_rand).degree < r_high/dis/np.pi*180,
                coord_neighbors_rand.separation(coord_rand).degree > r_low / dis / np.pi*180)]

            # make some cuts
            cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['MASS_MED'] > masscut_low]
            cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['MASS_MED'] < masscut_high]

            mass_neighbors_rand = cat_neighbors_rand['MASS_MED']
            if len(mass_neighbors_rand) != 0:
                if ssfq == 'all':
                    sfq_weights_rand = np.ones(len(mass_neighbors_rand))
                elif ssfq == 'ssf':
                    sfq_weights_rand = cat_neighbors_rand['sfProb']
                else:
                    sfq_weights_rand = 1 - cat_neighbors_rand['sfProb']

                weights = np.array(sfq_weights_rand / completeness_est(mass_neighbors_rand, cat_neighbors_rand['sfProb'], z))
                if not rel_scale:
                    counts_gals_rand = np.histogram(mass_neighbors_rand, weights=weights, bins=bin_edges)[0]
                else:
                    rel_mass_neighbors_rand = mass_neighbors_rand - mass_cen
                    counts_gals_rand = np.histogram(rel_mass_neighbors_rand, weights=weights, bins=rel_bin_edges)[0]

            else:
                counts_gals_rand += np.zeros(bin_number)

            n = n + 1

    return coord_rand_list, counts_gals_rand, flag_bkg


def bkg_2(cat_neighbors_z_slice_rand, coord_massive_gal_rand, mass_cen):
    counts_gals_rand = np.zeros(bin_number)
    flag_bkg2 = 0

    cat_neighbors_rand = cat_neighbors_z_slice_rand[abs(cat_neighbors_z_slice_rand['RA'] - coord_massive_gal_rand.ra) < 1 / dis / np.pi * 180]
    cat_neighbors_rand = cat_neighbors_rand[abs(cat_neighbors_rand['DEC'] - coord_massive_gal_rand.dec) < 1 / dis / np.pi * 180]
    coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)

    # exclude bkg apertures that contains galaxies more massive than central
    # if len(cat_neighbors_rand) != 0:
    #     if max(cat_neighbors_rand['MASS_MED']) > mass_cen:
    #         return 0,0,0

    # choose radial range
    cat_neighbors_rand = cat_neighbors_rand[
        np.logical_and(coord_neighbors_rand.separation(coord_massive_gal_rand).degree < (r_high + 0.1) / dis / np.pi * 180,
                       coord_neighbors_rand.separation(coord_massive_gal_rand).degree > r_high / dis / np.pi * 180)]

    # make some cuts
    cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['MASS_MED'] > masscut_low]
    cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['MASS_MED'] < masscut_high]

    mass_neighbors_rand = cat_neighbors_rand['MASS_MED']
    if len(mass_neighbors_rand) != 0:
        if ssfq == 'all':
            sfq_weights_rand = np.ones(len(mass_neighbors_rand))
        elif ssfq == 'ssf':
            sfq_weights_rand = cat_neighbors_rand['sfProb']
        else:
            sfq_weights_rand = 1 - cat_neighbors_rand['sfProb']

        weights = np.array(sfq_weights_rand / completeness_est(mass_neighbors_rand, cat_neighbors_rand['sfProb'], z))
        if not rel_scale:
            counts_gals_rand = np.histogram(mass_neighbors_rand, weights=weights, bins=bin_edges)[0]
        else:
            rel_mass_neighbors_rand = mass_neighbors_rand - mass_cen
            counts_gals_rand = np.histogram(rel_mass_neighbors_rand, weights=weights, bins=rel_bin_edges)[0]

    else:
        counts_gals_rand += np.zeros(bin_number)

    return counts_gals_rand, flag_bkg2


def correct_for_masked_area(ra, dec):
    # correct area for normalization if it is partially in masked region
    if not correct_masked:
        return np.ones(bin_number), np.ones(bin_number)
    else:
        cat_nomask = cat_random_nomask[abs(cat_random_nomask['RA'] - ra) < r_iso / dis / np.pi * 180]
        cat_nomask = cat_nomask[abs(cat_nomask['DEC'] - dec) < r_iso / dis / np.pi * 180]
        cat_nomask = cat_nomask[SkyCoord(cat_nomask['RA'] * u.deg, cat_nomask['DEC'] * u.deg).separation
                            (SkyCoord(ra * u.deg, dec * u.deg)).degree < r_iso / dis / np.pi * 180]

        cat_mask = cat_random[abs(cat_random['RA'] - ra) < r_iso / dis / np.pi * 180]
        cat_mask = cat_mask[abs(cat_mask['DEC'] - dec) < r_iso / dis / np.pi * 180]
        cat_mask = cat_mask[SkyCoord(cat_mask['RA'] * u.deg, cat_mask['DEC'] * u.deg).separation
                        (SkyCoord(ra * u.deg, dec * u.deg)).degree < r_iso / dis / np.pi * 180]

        if len(cat_nomask) == 0:
            return np.zeros(bin_number), np.zeros(bin_number)
        else:
            coord = SkyCoord(ra * u.deg, dec * u.deg)
            coord_nomask = SkyCoord(cat_nomask['RA'] * u.deg, cat_nomask['DEC'] * u.deg)
            radius_list_nomask = coord_nomask.separation(coord).degree / 180. * np.pi * dis * 1000
            count_nomask = np.histogram(radius_list_nomask, bins=bin_edges)[0]
            count_nomask = np.array(count_nomask).astype(float)
            if len(cat_mask) == 0:
                count_mask = np.zeros(bin_number)
            else:
                coord_mask = SkyCoord(cat_mask['RA'] * u.deg, cat_mask['DEC'] * u.deg)
                radius_list_mask = coord_mask.separation(coord).degree / 180. * np.pi * dis * 1000
                count_mask = np.histogram(radius_list_mask, bins=bin_edges)[0]
                count_mask = np.array(count_mask).astype(float)

            return count_mask, count_nomask


def cut_random_cat(cat_rand, coord_list):
    # cut random point catalog to avoid overlapping
    add_to_random_points(coord_list)
    for coord in coord_list:
        coord_rand_list = SkyCoord(cat_rand['RA'] * u.deg, cat_rand['DEC'] * u.deg)
        cat_rand = cat_rand[coord_rand_list.separation(coord).degree > r_iso * 2 / dis / np.pi * 180]
    return cat_rand


def add_to_random_points(coord_list):
    # store coord of selected random points just for record
    for coord in coord_list:
        cat_random_points.add_row([coord.ra.value, coord.dec.value, gal['NUMBER']])

    return 0


def completeness_est(mass_list, sfProb_list, z):
    try:
        completeness_sf = np.genfromtxt('../mass_completeness_data/allFields_' + str(round(z - 0.1, 1)) + '_z_' + str(round(z + 0.1,1)) + '_sf_nopert_nan.txt')
        completeness_q = np.genfromtxt('../mass_completeness_data/allFields_' + str(round(z - 0.1, 1)) + '_z_' + str(round(z + 0.1,1)) + '_q_nopert_nan.txt')
        completeness = np.array([])
        for idx in range(len(mass_list)):
            if sfProb_list[idx] > 0.5:
                completeness = np.append(completeness, np.interp(mass_list[idx], completeness_sf[0], completeness_sf[3]))
            else:
                completeness = np.append(completeness, np.interp(mass_list[idx], completeness_q[0], completeness_q[3]))

        completeness[np.isnan(completeness)] = 1.
        return completeness
    except:
        return np.ones(len(mass_list))


def scatter():
    z_scatter = np.random.normal(cat_gal[zkeyname], 0.044 * (cat_gal[zkeyname] + 1))  # photoz scatter
    mass_scatter = np.log10(abs(np.random.normal(10 ** (cat_gal['MASS_MED'] - 10),
                        (cat_gal['MASS_SUP'] - cat_gal['MASS_INF'])/2 * 10 ** (cat_gal['MASS_MED'] - 10)))) + 10  # mass scatter

    cat_gal['MASS_MED'] = mass_scatter
    cat_gal[zkeyname] = z_scatter


def check_cat(cat):
    for keyname in [zkeyname, 'MASS_MED', 'MASS_SUP', 'MASS_INF', 'preds_median', 'sfProb','NUMBER','RA','DEC']:
        if True in np.isnan(cat[keyname]):
            raise ValueError('NaN in necessary parameters: ' + keyname)
        elif True in np.isinf(cat[keyname]):
            raise ValueError('Inf in necessary parameters: ' + keyname)

# multi-threading settings
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nProcs = comm.Get_size()

# ################# START #####################
all_z = False
correct_masked = True
save_results = True
save_catalogs = False
rel_scale = False

masscut_low = 7.0
masscut_high = 13.0
masscut_host = 11.15
r_iso = 0.7 # Mpc (isolation criteria radius)
r_high = 0.7  # Mpc
r_low = 0.0  # Mpc
sat_z_cut = 4.5
csfq = 'all'  # csf, cq, all
ssfq = sys.argv[1]
boot_num = 5
zkeyname = 'ZPHOT'

# main loop
z = eval(sys.argv[2])
z_bin_size = 0.1

bin_number = 20
bin_edges = np.linspace(masscut_low, masscut_high, num=bin_number+1)
rel_bin_edges = np.linspace(-4, 0, num=bin_number+1)

catalog_path = 'catalogs/'
path = 'test_smf_new_cat/'
if path[-1] != '/' and save_results:
    raise NameError('path is not a directory!')
elif save_results:
    print('will save results to ' + path)
else:
    print('will NOT save results!')


# distribute data and collect results
if rank == 0:

    print('radius range:', r_low, r_high, 'sat_z_cut:', sat_z_cut)
    print('csfq =', csfq, 'ssfq =', ssfq, masscut_low, masscut_high)

    # set job distribution
    cat_names = ['COSMOS_deep', 'ELAIS_deep', 'XMM-LSS_deep', 'DEEP_deep', 'SXDS_uddd']
    if nProcs == len(cat_names)+1:
        for Proc in range(1, nProcs):
            comm.send(cat_names[Proc-1], dest=Proc)

    # stand-by to collect results
    smf_dist_tot = np.zeros(bin_number)
    smf_dist_bkg_tot = np.zeros(bin_number)
    smf_dist_sat_tot = np.zeros(bin_number)
    smf_dist_inf_tot = np.zeros(bin_number)
    smf_dist_bkg_inf_tot = np.zeros(bin_number)
    smf_dist_sat_inf_tot = np.zeros(bin_number)
    smf_dist_sup_tot = np.zeros(bin_number)
    smf_dist_bkg_sup_tot = np.zeros(bin_number)
    smf_dist_sat_sup_tot = np.zeros(bin_number)
    mass_cens_tot = []
    isolated_counts_tot = 0
    bkg_counts_tot = 0
    for i in range(1, nProcs):
        smf_dist, smf_dist_bkg, smf_dist_sat, mass_cens, isolated_counts, bkg_counts = comm.recv(source=MPI.ANY_SOURCE)

        smf_dist_tot += smf_dist[0]
        smf_dist_bkg_tot += smf_dist_bkg[0]
        smf_dist_sat_tot += smf_dist_sat[0]

        smf_dist_inf_tot += smf_dist[1]
        smf_dist_bkg_inf_tot += smf_dist_bkg[1]
        smf_dist_sat_inf_tot += smf_dist_sat[1]

        smf_dist_sup_tot += smf_dist[2]
        smf_dist_bkg_sup_tot += smf_dist_bkg[2]
        smf_dist_sat_sup_tot += smf_dist_sat[2]

        mass_cens_tot += mass_cens # addition of two lists (append)
        isolated_counts_tot += isolated_counts
        bkg_counts_tot += bkg_counts

    # output result to file
    print(round(sum(smf_dist_tot)), round(sum(smf_dist_bkg_tot)), round(sum(smf_dist_sat_tot)))
    smf_dist_cen_tot = np.histogram(mass_cens_tot, bins=bin_edges)[0]
    if len(mass_cens_tot) == isolated_counts_tot and save_results:
        filename = path + 'smf_' + str(r_low) + '_'+str(r_high) + '_' + str(masscut_low) + '_' + str(csfq) + '_' + str(ssfq) + '_' + str(round(z, 1))
        np.save(filename + '_total', [smf_dist_tot, smf_dist_inf_tot, smf_dist_sup_tot, isolated_counts_tot])
        np.save(filename + '_bkg', [smf_dist_bkg_tot, smf_dist_bkg_inf_tot, smf_dist_bkg_sup_tot, isolated_counts_tot])
        np.save(filename + '_sat', [smf_dist_sat_tot, smf_dist_sat_inf_tot, smf_dist_sat_sup_tot, isolated_counts_tot])

        if not rel_scale:
            np.save(path + 'bin_edges', bin_edges)
            np.save(filename + '_cen', [smf_dist_cen_tot, isolated_counts_tot])
        else:
            np.save(path + 'bin_edges', rel_bin_edges)

        print('total number', round(sum(smf_dist_tot)))
        print('total number in bkg', round(sum(smf_dist_bkg_tot)))
        print('massive counts:', isolated_counts_tot)
    elif not save_results:
        print('isolated counts', isolated_counts_tot,'bkg counts', bkg_counts_tot)
    else:
        print(len(mass_cens_tot), isolated_counts_tot, 'Warning: wrong numbers! (Results not saved)')

# calculations
else:

    # read in data catalog
    cat_name = comm.recv(source=MPI.ANY_SOURCE)
    cat_gal = Table.read(catalog_path + 's16a_' + cat_name + '_masterCat_newz.fits')
    check_cat(cat_gal)
    cat_gal = cat_gal[cat_gal[zkeyname] < 1.3]
    cat_gal = cat_gal[np.logical_and(cat_gal['preds_median'] < 0.89, cat_gal['inside'] == True)]
    cat_gal = cat_gal[cat_gal['MASS_MED'] > masscut_low]
    cat_random_points = Table(names=('RA', 'DEC', 'GAL_ID'))  # to store position of selected random apertures

    print('=========rank=' + str(rank) + '======z=' + str(round(z, 1)) + '=========='+cat_name+'===')

    # bootstrap resampling
    smf_dist_arr = np.zeros(bin_number)
    smf_dist_bkg_arr = np.zeros(bin_number)
    mass_key_ori = cat_gal['MASS_MED'].copy()
    z_key_ori = cat_gal[zkeyname].copy()
    mass_centrals_ori = []
    isolated_counts_ori = 0
    count_bkg_ori = 0
    for boot_iter in range(boot_num):
        if boot_iter != 0:
            cat_gal['MASS_MED'] = mass_key_ori
            cat_gal[zkeyname] = z_key_ori
            scatter()
            boot_idx = bootstrap(np.arange(len(cat_gal)), bootnum=1)
            cat_gal_copy = cat_gal[boot_idx[0].astype(int)]
        else:
            cat_gal_copy = cat_gal

        # select massive galaxies
        cat_massive_gal = cat_gal_copy[cat_gal_copy['MASS_MED'] > masscut_host]

        # read in random point catalog
        cat_random = Table.read(catalog_path + 's16a_' + cat_name + '_random.fits')
        try:
            cat_random = cat_random[cat_random['inside_u'] == True]
        except KeyError:
            cat_random = cat_random[cat_random['inside_uS'] == True]
        cat_random_nomask = np.copy(cat_random)
        cat_random = cat_random[cat_random['MASK'] == False]
        cat_random_points = Table(names=('RA', 'DEC', 'GAL_ID'))  # to store position of selected random apertures

        cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal[zkeyname] - z) < z_bin_size]
        coord_massive_gal = SkyCoord(cat_massive_z_slice['RA'] * u.deg, cat_massive_z_slice['DEC'] * u.deg)

        # ......
        isolated_counts = 0
        smf_dist = np.zeros(bin_number)
        smf_dist_bkg = np.zeros(bin_number)
        mass_centrals = []
        count_bkg = 0
        massive_count = 0
        print('massive gals:', len(cat_massive_z_slice))

        for gal in cat_massive_z_slice: # [np.random.randint(len(cat_massive_z_slice), size=300)]:
            massive_count += 1
            dis = WMAP9.angular_diameter_distance(gal[zkeyname]).value
            coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)

            # prepare neighbors catalog
            cat_neighbors_z_slice = cat_gal_copy[abs(cat_gal_copy[zkeyname] - gal[zkeyname]) < sat_z_cut * 0.044 * (1 + gal[zkeyname])]
            cat_neighbors = cat_neighbors_z_slice[abs(cat_neighbors_z_slice['RA'] - gal['RA']) < r_iso / dis / np.pi * 180]
            cat_neighbors = cat_neighbors[abs(cat_neighbors['DEC'] - gal['DEC']) < r_iso / dis / np.pi * 180]

            # ## ## spatial selection
            if len(cat_neighbors) == 0:  # central gals which has no companion
                continue

            else:
                # choose sats within r_high
                ind = KDTree(np.array(cat_neighbors['RA', 'DEC']).tolist()).query_radius([(gal['RA'], gal['DEC'])], r_iso / dis / np.pi * 180)
                cat_neighbors = cat_neighbors[ind[0]]
                cat_neighbors = cat_neighbors[cat_neighbors['NUMBER'] != gal['NUMBER']]

                # isolation cut on central
                if len(cat_neighbors) == 0:  # central gals which has no companion
                    continue
                elif gal['MASS_MED'] < max(cat_neighbors['MASS_MED']):  # no more-massive companions
                    continue

                # choose sats within r_high
                ind2 = KDTree(np.array(cat_neighbors['RA', 'DEC']).tolist()).query_radius([(gal['RA'], gal['DEC'])],r_high / dis / np.pi * 180)
                cat_neighbors = cat_neighbors[ind2[0]]
                cat_neighbors = cat_neighbors[cat_neighbors['NUMBER'] != gal['NUMBER']]
                if len(cat_neighbors) == 0:  # central gals which has no companion
                    continue

                # exclude sats within r_low
                ind3 = KDTree(np.array(cat_neighbors['RA', 'DEC']).tolist()).query_radius([(gal['RA'], gal['DEC'])], r_low / dis / np.pi * 180)
                cat_neighbors = np.delete(cat_neighbors, ind3[0], axis=0)

                cat_neighbors = cat_neighbors[cat_neighbors['NUMBER'] != gal['NUMBER']]
                if len(cat_neighbors) == 0:  # central gals which has no companion
                    continue

            # cut on central SF/Q
            if csfq == 'csf' and gal['sfProb'] < 0.5:
                continue
            elif csfq == 'cq' and gal['sfProb'] >= 0.5:
                continue

            # cut on companion sample (cut the final sample)
            cat_neighbors = cat_neighbors[cat_neighbors['MASS_MED'] > masscut_low]
            cat_neighbors = cat_neighbors[cat_neighbors['MASS_MED'] < masscut_high]
            mass_neighbors = cat_neighbors['MASS_MED']
            if len(cat_neighbors) == 0:  # central gals which has no companion
                continue

            # Core Function: statistics #
            isolated_counts += 1
            mass_centrals.append(gal['MASS_MED'])

            if ssfq == 'all':
                sfq_weights = np.ones(len(cat_neighbors))
            elif ssfq == 'ssf':
                sfq_weights = cat_neighbors['sfProb']
            else:
                sfq_weights = 1 - cat_neighbors['sfProb']

            # absolute / relative mass scale
            sat_weights = np.array(sfq_weights/completeness_est(mass_neighbors, cat_neighbors['sfProb'], z))
            if not rel_scale:
                count_binned = np.histogram(mass_neighbors, weights=sat_weights, bins=bin_edges)[0]
            else:
                rel_mass_neighbors = mass_neighbors - gal['MASS_MED']
                count_binned = np.histogram(rel_mass_neighbors, weights=sat_weights, bins=rel_bin_edges)[0]

            sat_counts = np.array(count_binned, dtype='f8')
            smf_dist += sat_counts

            coord_random_list, sat_bkg, flag_bkg = bkg(cat_neighbors_z_slice, coord_massive_gal, gal['MASS_MED'])
            if flag_bkg == 0:
                cat_random = cut_random_cat(cat_random, coord_random_list)
                # sat_bkg, flag_bkg = bkg_2(cat_neighbors_z_slice, coord_massive_gal, gal['MASS_MED'])
                smf_dist_bkg += sat_bkg
                count_bkg += 1
            else:
                flag_bkg = 0

        # add results from this bootstrap iteration
        smf_dist_bkg = smf_dist_bkg / float(count_bkg) * isolated_counts
        smf_dist_arr = np.vstack((smf_dist_arr, smf_dist))
        smf_dist_bkg_arr = np.vstack((smf_dist_bkg_arr, smf_dist_bkg))
        if boot_iter == 0:
            mass_centrals_ori = mass_centrals
            isolated_counts_ori = isolated_counts
            count_bkg_ori = count_bkg


    # combine bootstrap results and calculate error
    smf_dist_arr = smf_dist_arr[1:]
    smf_dist_bkg_arr = smf_dist_bkg_arr[1:]
    smf_dist_sat_arr = smf_dist_arr - smf_dist_bkg_arr

    smf_dist_avg = np.average(smf_dist_arr, axis=0)
    smf_dist_bkg_avg = np.average(smf_dist_bkg_arr, axis=0)
    smf_dist_sat_avg = np.average(smf_dist_sat_arr, axis=0)

    smf_dist_inf = np.percentile(smf_dist_arr, 16, axis=0)
    smf_dist_bkg_inf = np.percentile(smf_dist_bkg_arr, 16, axis=0)
    smf_dist_sat_inf = np.percentile(smf_dist_sat_arr, 16, axis=0)

    smf_dist_sup = np.percentile(smf_dist_arr, 84, axis=0)
    smf_dist_bkg_sup = np.percentile(smf_dist_bkg_arr, 84, axis=0)
    smf_dist_sat_sup = np.percentile(smf_dist_sat_arr, 84, axis=0)

    smf_dist_avg_error = [smf_dist_avg, smf_dist_inf, smf_dist_sup]
    smf_dist_bkg_avg_error = [smf_dist_bkg_avg, smf_dist_bkg_inf, smf_dist_bkg_sup]
    smf_dist_sat_avg_error = [smf_dist_sat_avg, smf_dist_sat_inf, smf_dist_sat_sup]

    # cat_random_points.write('smf_random_points_' + cat_name + '_' + str(z) + '.fits', overwrite=True)
    comm.send((smf_dist_avg_error, smf_dist_bkg_avg_error, smf_dist_sat_avg_error, mass_centrals_ori, isolated_counts_ori,count_bkg_ori), dest=0)