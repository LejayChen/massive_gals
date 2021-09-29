import sys
from random import random
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.cosmology import WMAP9
from astropy.table import Table
from sklearn.neighbors import KDTree
from mpi4py import MPI


def bkg(cat_neighbors_z_slice_rand, coord_massive_gal_rand, mass_cen):
    global z
    counts_gals_rand = 0
    n = 0
    num_before_success = 0
    flag_bkg = 0

    coord_rand_list = []
    while n < 1:  # get several blank pointing's to estimate background
        id_rand = int(random() * len(cat_random_copy))
        ra_rand = cat_random_copy[id_rand]['RA']
        dec_rand = cat_random_copy[id_rand]['DEC']
        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(ra_rand, dec_rand, unit="deg"), coord_massive_gal_rand, nthneighbor=1)

        num_before_success += 1
        if num_before_success > 100:
            flag_bkg = 1
            break

        if sep2d.degree > 1.4/dis/np.pi*180:  # make sure the random pointing is away from any central galaxy (blank)
            num_before_success = 0
            coord_rand = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)
            coord_rand_list.append(coord_rand)
            cat_neighbors_rand = cat_neighbors_z_slice_rand[abs(cat_neighbors_z_slice_rand['RA'] - ra_rand) < 0.7/dis/np.pi*180]
            cat_neighbors_rand = cat_neighbors_rand[abs(cat_neighbors_rand['DEC'] - dec_rand) < 0.7/dis/np.pi*180]
            coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)

            # exclude bkg apertures that contains galaxies more massive than central
            if len(cat_neighbors_rand) != 0:
                if max(cat_neighbors_rand[mass_keyname]) < mass_cen:
                    num_before_success = 0
                else:
                    continue

            # choose radial range
            cat_neighbors_rand = cat_neighbors_rand[np.logical_and(coord_neighbors_rand.separation(coord_rand).degree < r_high/dis/np.pi*180,
                coord_neighbors_rand.separation(coord_rand).degree > r_low / dis / np.pi*180)]

            # make some cuts
            cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand[mass_keyname] > masscut_low]
            cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand[mass_keyname] < masscut_high]

            mass_neighbors_rand = cat_neighbors_rand[mass_keyname]
            if len(mass_neighbors_rand) != 0:
                if ssfq == 'all':
                    sfq_weights_rand = np.ones(len(mass_neighbors_rand))
                elif ssfq == 'ssf':
                    sfq_weights_rand = cat_neighbors_rand['sfProb']
                else:
                    sfq_weights_rand = 1 - cat_neighbors_rand['sfProb']

                weights = np.array(sfq_weights_rand / completeness_est(mass_neighbors_rand, cat_neighbors_rand['sfProb'], z))
                counts_gals_rand = np.histogram(mass_neighbors_rand, weights=weights, bins=bin_edges)[0]

            else:
                counts_gals_rand += np.zeros(bin_number)

            n = n + 1

    return coord_rand_list, counts_gals_rand, flag_bkg


def correct_for_masked_area(ra, dec):
    # correct area for normalization if it is partially in masked region
    if not correct_masked:
        return np.ones(bin_number), np.ones(bin_number)
    else:
        cat_nomask = cat_random_nomask[abs(cat_random_nomask['RA'] - ra) < 0.7 / dis / np.pi * 180]
        cat_nomask = cat_nomask[abs(cat_nomask['DEC'] - dec) < 0.7 / dis / np.pi * 180]
        cat_nomask = cat_nomask[SkyCoord(cat_nomask['RA'] * u.deg, cat_nomask['DEC'] * u.deg).separation
                            (SkyCoord(ra * u.deg, dec * u.deg)).degree < 0.7 / dis / np.pi * 180]

        cat_mask = cat_random[abs(cat_random['RA'] - ra) < 0.7 / dis / np.pi * 180]
        cat_mask = cat_mask[abs(cat_mask['DEC'] - dec) < 0.7 / dis / np.pi * 180]
        cat_mask = cat_mask[SkyCoord(cat_mask['RA'] * u.deg, cat_mask['DEC'] * u.deg).separation
                        (SkyCoord(ra * u.deg, dec * u.deg)).degree < 0.7 / dis / np.pi * 180]

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
    for coord in coord_list:
        coord_rand_list = SkyCoord(cat_rand['RA'] * u.deg, cat_rand['DEC'] * u.deg)
        cat_rand = cat_rand[coord_rand_list.separation(coord).degree > 1.05 / dis / np.pi * 180]
    return cat_rand


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


# multi-threading settings
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nProcs = comm.Get_size()

# ################# START #####################
all_z = False
correct_masked = True
save_results = True

cat_name = sys.argv[1]
z_keyname = 'zKDEPeak'
mass_keyname = 'MASS_MED'
masscut_low = 7.0
masscut_high = 13.0
masscut_host = 11.15
r_high = 0.7  # Mpc
r_low = 0.5  # Mpc
sat_z_cut = 4.5
csfq = 'all'  # csf, cq, all
ssfq = sys.argv[2]

catalog_path = 'catalogs/'
path = 'test_smf/'

bin_number = 20
bin_edges = np.linspace(masscut_low, masscut_high, num=bin_number+1)

if path[-1] != '/' and save_results:
    raise NameError('path is not a directory!')
elif save_results:
    print('will save results to ' + path)
else:
    print('will NOT save results!')

# read in data catalog
if cat_name == 'SXDS_uddd':
    cat_name = 'SXDS3_uddd'
    cat = Table.read(catalog_path+'s16a_' + cat_name + '_masterCat.fits')
else:
    cat = Table.read(catalog_path+'CUT3_' + cat_name + '.fits')

cat = cat[cat[z_keyname] < 1.3]
cat_gal = cat[np.logical_and(cat['preds_median'] < 0.89, cat['inside'] == True)]
cat_gal = cat_gal[cat_gal[mass_keyname] > masscut_low]
cat_massive_gal = cat_gal[cat_gal[mass_keyname] > masscut_host]

# read-in random point catalog
cat_random = Table.read(catalog_path+'s16a_' + cat_name + '_random.fits')
try:
    cat_random = cat_random[cat_random['inside_u'] == True]
except KeyError:
    cat_random = cat_random[cat_random['inside_uS'] == True]
cat_random_nomask = np.copy(cat_random)
cat_random = cat_random[cat_random['MASK'] == False]
cat_random_points = Table(names=('RA', 'DEC', 'GAL_ID'))   # to store position of selected random apertures

if cat_name == 'SXDS3_uddd':
    cat_name = 'SXDS_uddd'

# main loop
z = eval(sys.argv[3])
z_bin_size = 0.1

print('=========rank='+str(rank)+'======z='+str(round(z, 1))+'==========')
print(csfq, ssfq, masscut_low, masscut_high)

cat_random_copy = np.copy(cat_random)  # reset random points catalog at each redshift
cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal[z_keyname] - z) < z_bin_size]
coord_massive_gal = SkyCoord(cat_massive_z_slice['RA'] * u.deg, cat_massive_z_slice['DEC'] * u.deg)
cat_all_z_slice = cat_gal

if rank == 0:

    # collect results
    smf_dist_tot = np.zeros(bin_number)
    smf_dist_bkg_tot = np.zeros(bin_number)
    isolated_counts_tot = 0
    for i in range(1, nProcs):
        smf_dist, smf_dist_bkg, isolated_counts = comm.recv(source=MPI.ANY_SOURCE)
        smf_dist_tot += smf_dist
        smf_dist_bkg_tot += smf_dist_bkg * isolated_counts
        isolated_counts_tot += isolated_counts

    # errorbars and processing of results
    smf_dist_sat_tot = smf_dist_tot - smf_dist_bkg_tot




    # output result to file
    filename = path + 'smf_' + str(r_low) + '_' + cat_name + '_' + \
                   str(masscut_low) + '_' + str(csfq) + '_' + str(ssfq) + '_' + str(round(z, 1))

    np.save(filename + '_total', [smf_dist_tot, isolated_counts_tot])
    np.save(filename + '_bkg', [smf_dist_bkg_tot, isolated_counts_tot])
    np.save(filename + '_sat', [smf_dist_sat_tot, isolated_counts_tot])
    np.save(path + 'bin_edges', bin_edges)
    print('total number', round(sum(smf_dist_tot)))
    print('total number in bkg', round(sum(ssmf_dist_bkg_tot)))
    print('massive counts:', len(cat_massive_z_slice), isolated_counts_tot)

else:
    nEach = len(cat_massive_z_slice) // (nProcs - 1)
    if rank == nProcs - 1:
        cat_massive_z_slice_each = cat_massive_z_slice[(rank-1) * nEach:]
    else:
        cat_massive_z_slice_each = cat_massive_z_slice[(rank-1) * nEach:(rank-1) * nEach + nEach]

    isolated_counts = 0
    smf_dist = np.zeros(bin_number)
    smf_dist_bkg = np.zeros(bin_number)
    count_bkg = 0
    for gal in cat_massive_z_slice_each:
        dis = WMAP9.angular_diameter_distance(gal[z_keyname]).value
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)

        # prepare neighbors catalog
        cat_neighbors_z_slice = cat_all_z_slice[abs(cat_all_z_slice[z_keyname] - gal[z_keyname]) < sat_z_cut * 0.044 * (1 + gal[z_keyname])]
        cat_neighbors = cat_neighbors_z_slice[abs(cat_neighbors_z_slice['RA'] - gal['RA']) < 0.7 / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[abs(cat_neighbors['DEC'] - gal['DEC']) < 0.7 / dis / np.pi * 180]

        # ## ## spatial selection
        if len(cat_neighbors) == 0:  # central gals which has no companion
            continue

        else:
            # choose sats within r_high
            ind = KDTree(np.array(cat_neighbors['RA', 'DEC']).tolist()).query_radius([(gal['RA'], gal['DEC'])], r_high / dis / np.pi * 180)
            cat_neighbors = cat_neighbors[ind[0]]

            # isolation cut on central
            if gal[mass_keyname] < max(cat_neighbors[mass_keyname]):  # no more-massive companions
                continue

            # exclude sats within r_low
            ind2 = KDTree(np.array(cat_neighbors['RA', 'DEC']).tolist()).query_radius([(gal['RA'], gal['DEC'])], r_low / dis / np.pi * 180)
            cat_neighbors = np.delete(cat_neighbors, ind2[0], axis=0)

            cat_neighbors = cat_neighbors[cat_neighbors['NUMBER'] != gal['NUMBER']]
            if len(cat_neighbors) == 0:  # central gals which has no companion
                continue

        # cut on central SF/Q
        if csfq == 'csf' and gal['sfProb'] < 0.5:
            continue
        elif csfq == 'cq' and gal['sfProb'] >= 0.5:
            continue

        # cut on companion sample (cut the final sample)
        cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] > masscut_low]
        cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] < masscut_high]
        mass_neighbors = cat_neighbors[mass_keyname]
        if len(cat_neighbors) == 0:  # central gals which has no companion
            continue

        # Core Function: statistics #
        isolated_counts += 1
        if ssfq == 'all':
            sfq_weights = np.ones(len(cat_neighbors))
        elif ssfq == 'ssf':
            sfq_weights = cat_neighbors['sfProb']
        else:
            sfq_weights = 1 - cat_neighbors['sfProb']

        count_binned = np.histogram(mass_neighbors,
                                    weights=np.array(sfq_weights/completeness_est(mass_neighbors, cat_neighbors['sfProb'], z)),
                                    bins=bin_edges)[0]

        sat_counts = np.array(count_binned, dtype='f8')
        smf_dist += sat_counts

        flag_bkg = 0
        if flag_bkg == 0:
            coord_random_list, sat_bkg, flag_bkg = bkg(cat_neighbors_z_slice, coord_massive_gal, gal[mass_keyname])
            cat_random_copy = cut_random_cat(cat_random_copy, coord_random_list)
            smf_dist_bkg += sat_bkg
            count_bkg += 1

    smf_dist_bkg = smf_dist_bkg/float(count_bkg)
    comm.send((smf_dist, smf_dist_bkg, isolated_counts), dest=0)

