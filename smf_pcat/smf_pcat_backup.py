import sys, os
from random import random
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.cosmology import *
from astropy.stats import bootstrap
from astropy.table import Table
from mpi4py import MPI

def check_edge(ra_rand, dec_rand):
    cat_random_cut = cat_random[abs(cat_random['ra'] - ra_rand) < r_iso / dis / np.pi * 180]
    cat_random_cut = cat_random_cut[abs(cat_random_cut['dec'] - dec_rand) < r_iso / dis / np.pi * 180]
    try:
        ra_ratio = len(cat_random_cut[cat_random_cut['ra']<ra_rand])/len(cat_random_cut[cat_random_cut['ra']>ra_rand])
        dec_ratio = len(cat_random_cut[cat_random_cut['dec']<dec_rand])/len(cat_random_cut[cat_random_cut['dec']>dec_rand])
    except (ValueError, ZeroDivisionError):
        return True

    if ra_ratio > 1/0.75 or ra_ratio < 0.75:
        return True
    elif dec_ratio > 1/0.75 or dec_ratio < 0.75:
        return True
    else:
        return False


def bkg(cat_neighbors_z_slice_rand, coord_massive_gal_rand, mass_cen):
    cat_neighbors_z_slice_rand = cat_neighbors_z_slice_rand[cat_neighbors_z_slice_rand[mass_keyname] > masscut_low]
    cat_neighbors_z_slice_rand = cat_neighbors_z_slice_rand[cat_neighbors_z_slice_rand[mass_keyname] < masscut_high]
    global z, overlapping_factor
    counts_gals_rand = np.zeros(bin_number)
    n = 0
    num_before_success = 0
    flag_bkg = 0

    coord_rand_list = []
    while n < 1:  # get several blank pointing's to estimate background
        id_rand = int(random() * len(cat_random))
        ra_rand = cat_random[id_rand]['ra']
        dec_rand = cat_random[id_rand]['dec']
        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(ra_rand, dec_rand, unit="deg"), coord_massive_gal_rand, nthneighbor=1)

        num_before_success += 1
        if num_before_success > 50:
            flag_bkg = 1
            break

        if sep2d.degree > overlapping_factor*2*r_iso/dis/np.pi*180:  # make sure the random pointing is away from any central galaxy (blank)
            if check_edge(ra_rand, dec_rand):
                continue

            coord_rand = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)
            coord_rand_list.append(coord_rand)
            cat_neighbors_rand = cat_neighbors_z_slice_rand[abs(cat_neighbors_z_slice_rand['RA'] - ra_rand) < r_iso*3/dis/np.pi*180]
            cat_neighbors_rand = cat_neighbors_rand[abs(cat_neighbors_rand['DEC'] - dec_rand) < r_iso*3/dis/np.pi*180]
            coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)
            if len(cat_neighbors_rand) == 0:
                continue

            # choose radial range
            cat_neighbors_rand = cat_neighbors_rand[np.logical_and(coord_neighbors_rand.separation(coord_rand).degree < r_high/dis/np.pi*180,
                coord_neighbors_rand.separation(coord_rand).degree > r_low / dis / np.pi*180)]

            mass_neighbors_rand = cat_neighbors_rand[mass_keyname]
            if len(cat_neighbors_rand) == 0:
                continue
                # counts_gals_rand += np.zeros(bin_number)
            # elif max(mass_neighbors_rand)>masscut_host:
            #     continue
            else:
                if ssfq == 'all':
                    sfq_weights_rand = np.ones(len(mass_neighbors_rand))
                elif ssfq == 'ssf':
                    sfq_weights_rand = cat_neighbors_rand[sfq_keyname]
                else:
                    sfq_weights_rand = 1 - cat_neighbors_rand[sfq_keyname]

                # weights = np.array(sfq_weights_rand / completeness_est(mass_neighbors_rand, cat_neighbors_rand[sfq_keyname], z))
                weights = np.array(sfq_weights_rand)
                if not rel_scale:
                    counts_gals_rand = np.histogram(mass_neighbors_rand, weights=weights, bins=bin_edges)[0]
                else:
                    rel_mass_neighbors_rand = mass_neighbors_rand - mass_cen
                    counts_gals_rand = np.histogram(rel_mass_neighbors_rand, weights=weights, bins=rel_bin_edges)[0]
                
            n = n + 1

    return coord_rand_list, counts_gals_rand, flag_bkg


def correct_for_masked_area(ra, dec):
    # correct area for normalization if it is partially in masked region
    if not correct_masked:
        return np.ones(bin_number), np.ones(bin_number)
    else:
        cat_nomask = cat_random_nomask[abs(cat_random_nomask['ra'] - ra) < r_iso / dis / np.pi * 180]
        cat_nomask = cat_nomask[abs(cat_nomask['dec'] - dec) < r_iso / dis / np.pi * 180]
        cat_nomask = cat_nomask[SkyCoord(cat_nomask['ra'] * u.deg, cat_nomask['dec'] * u.deg).separation
                            (SkyCoord(ra * u.deg, dec * u.deg)).degree < r_iso / dis / np.pi * 180]

        cat_mask = cat_random[abs(cat_random['ra'] - ra) < r_iso / dis / np.pi * 180]
        cat_mask = cat_mask[abs(cat_mask['dec'] - dec) < r_iso / dis / np.pi * 180]
        cat_mask = cat_mask[SkyCoord(cat_mask['ra'] * u.deg, cat_mask['dec'] * u.deg).separation
                        (SkyCoord(ra * u.deg, dec * u.deg)).degree < r_iso / dis / np.pi * 180]

        if len(cat_nomask) == 0:
            return np.zeros(bin_number), np.zeros(bin_number)
        else:
            coord = SkyCoord(ra * u.deg, dec * u.deg)
            coord_nomask = SkyCoord(cat_nomask['ra'] * u.deg, cat_nomask['dec'] * u.deg)
            radius_list_nomask = coord_nomask.separation(coord).degree / 180. * np.pi * dis * 1000
            count_nomask = np.histogram(radius_list_nomask, bins=bin_edges)[0]
            count_nomask = np.array(count_nomask).astype(float)
            if len(cat_mask) == 0:
                count_mask = np.zeros(bin_number)
            else:
                coord_mask = SkyCoord(cat_mask['ra'] * u.deg, cat_mask['dec'] * u.deg)
                radius_list_mask = coord_mask.separation(coord).degree / 180. * np.pi * dis * 1000
                count_mask = np.histogram(radius_list_mask, bins=bin_edges)[0]
                count_mask = np.array(count_mask).astype(float)

            return count_mask, count_nomask


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

# cosmology ##############
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# ################# START #####################
all_z = False
correct_masked = True
save_results = True
save_catalogs = False
rel_scale = False # relative scale


ssfq = sys.argv[1]
z_min = eval(sys.argv[2])
z_max= eval(sys.argv[3])
path = sys.argv[4]
bin_number = eval(sys.argv[5])
boot_num = int(eval(sys.argv[6]))

sigma_z = 0.06
masscut_low = eval(sys.argv[7])
masscut_high = 13.0
masscut_host = eval(sys.argv[8])
bin_edges = np.linspace(masscut_low, masscut_high, num=bin_number+1)
rel_bin_edges = np.linspace(-4, 0, num=bin_number+1)

r_iso = 1.0  # Mpc (isolation criteria radius)
r_high = 1.0  # Mpc
r_low = 0.0  # Mpc
sat_z_cut = 3.0
overlapping_factor = eval(sys.argv[9]) # closest distance factor a random position is to a massive galaxy (1=r_iso)
csfq = sys.argv[10]  # csf, cq, all

if path[-1] != '/' and save_results:
    raise NameError('path is not a directory!')
elif save_results:
    print('will save results to ' + path)
    if not os.path.exists(path):
        os.system('mkdir '+path)
else:
    print('will NOT save results!')


cat_names = ['COSMOS_deep','DEEP_deep','ELAIS_deep','XMM-LSS_deep']
cat_name = cat_names[rank % len(cat_names)]
cores_per_cat = nProcs // len(cat_names)
rank_in_cat = rank // len(cat_names)

catalog_path = '/home/lejay/projects/def-sawicki/lejay/phys_added_pcats/'
cat = Table.read(catalog_path + cat_name.replace('_deep','') + '_galaxies_241214_trimmed.fits')  # read-in
zkeyname = 'Z_COMBINE'
mass_keyname = 'MASS_MED'
sfq_keyname = 'sfq_nuvrk_balanced'
id_keyname = 'ID'
ra_key = 'RA'
dec_key = 'DEC'

if 'XMM' not in cat_name:
    cat = cat[cat['inside_u'] == True]
else:
    cat = cat[cat['inside_uS_deep'] == True]
    cat = cat[cat['inside_j'] == True]
cat = cat[cat['inside_hsc'] == True]
cat = cat[cat['isStar']==False]
cat = cat[cat['inside_hsc'] == True]
cat = cat[cat['isCompact']==False]
cat = cat[cat['i_compact_flag']==False]
cat = cat[cat['snr_i']>3]
cat = cat[cat['snr_r']>1.5]
cat = cat[cat['snr_z']>1.5]
cat_gal = cat[cat['i_cmodel']>0]
cat_gal = cat_gal[cat_gal[zkeyname] < 2.0]
cat_gal = cat_gal[cat_gal[sfq_keyname] >= 0.0]
cat_gal = cat_gal[cat_gal[sfq_keyname] <= 1.0]
cat_gal = cat_gal[cat_gal[mass_keyname] > masscut_low]
print(z_min,z_max)
print(cat_name,rank_in_cat,len(cat_gal))
del cat

# bootstrap resampling
smf_dist_arr = np.zeros(bin_number)
smf_dist_bkg_arr = np.zeros(bin_number)
mass_key_ori = cat_gal[mass_keyname].copy()  # original
z_key_ori = cat_gal[zkeyname].copy()  # original
mass_centrals_ori = []
isolated_counts_ori = 0
count_bkg_ori = 0
for boot_iter in range(boot_num):
    print('')
    print('boot_iter '+str(boot_iter)+' started.')
    
    if boot_iter > 0:  # number of iteration of bootstrapping
        cat_gal[mass_keyname] = mass_key_ori
        cat_gal[zkeyname] = z_key_ori
        z_scatter = np.random.normal(cat_gal[zkeyname], 0.06 * (cat_gal[zkeyname] + 1))  # photoz scatter
        mass_scatter = np.log10(abs(np.random.normal(10 ** (cat_gal[mass_keyname] - 10),
                          (cat_gal['MASS_SUP'] - cat_gal['MASS_INF'])/2 * 10 ** (cat_gal['MASS_MED'] - 10)))) + 10  # mass scatter

        # cat_gal[mass_keyname] = mass_scatter
        # cat_gal[zkeyname] = z_scatter
        boot_idx = bootstrap(np.arange(len(cat_gal)), bootnum=1)
        cat_gal_copy = cat_gal[boot_idx[0].astype(int)]
    else:
        print('original, no bootstrapping')
        cat_gal_copy = cat_gal

    # select massive galaxies
    cat_massive_gal = cat_gal_copy[cat_gal_copy[mass_keyname] > masscut_host]
    cat_massive_z_slice = cat_massive_gal[cat_massive_gal[zkeyname]>z_min]
    cat_massive_z_slice = cat_massive_z_slice[cat_massive_z_slice[zkeyname]<z_max]
    # cat_massive_z_slice = cat_massive_z_slice[:int(len(cat_massive_z_slice)/10)] # only use part of massive gals for test
    coord_massive_gal = SkyCoord(cat_massive_z_slice['RA'] * u.deg, cat_massive_z_slice['DEC'] * u.deg)
    print('massive gals rank_in_cat:', rank_in_cat, len(cat_massive_z_slice))

    # select isolated massive galaxies

    # split massive gal sample 
    if rank_in_cat != cores_per_cat -1:
        cat_massive_z_slice = cat_massive_z_slice[rank_in_cat*int(len(cat_massive_z_slice)/(nProcs/len(cat_names))) : (rank_in_cat+1)*int(len(cat_massive_z_slice)  /(nProcs/len(cat_names)))]
    else:
        cat_massive_z_slice = cat_massive_z_slice[rank_in_cat*int(len(cat_massive_z_slice)/(nProcs/len(cat_names))) : (rank_in_cat+1)*int(len(cat_massive_z_slice)  /(nProcs/len(cat_names)))]
    
    print('===rank:'+str(rank)+'===rank_in_cat=' + str(rank_in_cat) + '===z_min=' + str(round(z_min, 1)) + '===z_max=' + str(round(z_max, 1))+ '====='+cat_name+'==='+str(len(cat_massive_z_slice))+'===')
    # read in random point catalog
    cat_random = Table.read('/home/lejay/random_point_cat/'+cat_name + '_random_point.fits')
    cat_random = cat_random[cat_random['inside_hsc'] == 0]

    if cat_name == 'XMM-LSS_deep':
        cat_random = cat_random[cat_random['inside_uS_deep'] == 0]
        cat_random = cat_random[cat_random['inside_j'] == 0]
    else:
        cat_random = cat_random[cat_random['inside_u'] == 0]

    cat_random_nomask = np.copy(cat_random)
    cat_random = cat_random[cat_random['MASK'] != 0]

    # CORE CALCULATION
    cat_random_points = Table(names=('RA', 'DEC', 'GAL_ID'))  # to store position of selected random apertures
    isolated_counts = 0
    smf_dist = np.zeros(bin_number)
    smf_dist_bkg = np.zeros(bin_number)
    mass_centrals = []
    count_bkg = 0
    
    for gal in cat_massive_z_slice:  # [np.random.randint(len(cat_massive_z_slice), size=300)]:
        dis = cosmo.angular_diameter_distance(gal[zkeyname]).value
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)

        # prepare neighbors catalog
        cat_neighbors_z_slice = cat_gal_copy[abs(cat_gal_copy[zkeyname] - gal[zkeyname]) < sat_z_cut * 0.06 * (1 + gal[zkeyname])]
        cat_neighbors = cat_neighbors_z_slice[abs(cat_neighbors_z_slice['RA'] - gal['RA']) < 3 * r_iso / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[abs(cat_neighbors['DEC'] - gal['DEC']) < 3 * r_iso / dis / np.pi * 180]

        # #### spatial selection
        if len(cat_neighbors) == 0:  # central gals which has no companion
            continue
        else:
            # choose sats within r_high
            coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
            cat_neighbors = cat_neighbors[coord_neighbors.separation(coord_gal).degree < r_iso / dis / np.pi * 180]

            # isolation cut on central
            if len(cat_neighbors) == 0:  # central gals which has no companion
                continue
            elif gal[mass_keyname] < max(cat_neighbors[mass_keyname]):  # no more-massive companions
                continue
            else:
                pass
                # print(gal[mass_keyname], max(cat_neighbors[mass_keyname]),'1st')

            # choose sats within r_high
            coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
            cat_neighbors = cat_neighbors[coord_neighbors.separation(coord_gal).degree < r_high / dis / np.pi * 180]
            if len(cat_neighbors) == 0:  # central gals which has no companion
                continue

            # exclude sats within r_low
            coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
            if r_low > 0.0:
                cat_neighbors = cat_neighbors[coord_neighbors.separation(coord_gal).degree >= r_low / dis / np.pi * 180]
                if len(cat_neighbors) == 0:  # central gals which has no companion
                    continue

        # cut on central SF/Q
        if csfq == 'csf' and gal[sfq_keyname] < 0.5:
            continue
        elif csfq == 'cq' and gal[sfq_keyname] >= 0.5:
            continue

        # cut on companion sample (cut the final sample)
        cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] > masscut_low]
        cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] < masscut_high]
        mass_neighbors = cat_neighbors[mass_keyname]

        if len(cat_neighbors) == 0:  # central gals which has no companion
            continue

        # statistics #
        isolated_counts += 1
        mass_centrals.append(gal[mass_keyname])

        if ssfq == 'all':
            sfq_weights = np.ones(len(cat_neighbors))
        elif ssfq == 'ssf':
            sfq_weights = cat_neighbors[sfq_keyname]
        else:
            sfq_weights = 1 - cat_neighbors[sfq_keyname]

        # cylinder volume
        # z_sat_high = gal[zkeyname] + sat_z_cut * 0.06 * (1 + gal[zkeyname])
        # z_sat_low = gal[zkeyname] - sat_z_cut * 0.06 * (1 + gal[zkeyname])

        # z_test_bin_size = (z_sat_high-z_sat_low)/5
        # # volume_cylinder = 0
        # for z_test in np.arange(z_sat_low,z_sat_high,z_test_bin_size):
        #     volume = cosmo.comoving_volume(z_test+z_test_bin_size).value - cosmo.comoving_volume(z_test).value  # in Mpc^3
        #     radius_radians = r_iso/ cosmo.angular_diameter_distance(z_test).value # (in radian) radius at redshift of the central galaxy 
        #     solid_angle = np.pi*radius_radians**2 # in steraidans
        #     total_solid_angle = 4*np.pi # in steraidans
        #     volume_cylinder += volume*solid_angle/total_solid_angle
        # print('volume_cylinder', volume_cylinder)

        
        # sat_weights = np.array(sfq_weights/completeness_est(mass_neighbors, cat_neighbors[sfq_keyname], z))
        sat_weights = np.array(sfq_weights)

        #absolute / relative mass scale
        if not rel_scale:
            count_binned = np.histogram(mass_neighbors, weights=sat_weights, bins=bin_edges)[0]
        else:
            rel_mass_neighbors = mass_neighbors - gal[mass_keyname]
            count_binned = np.histogram(rel_mass_neighbors, weights=sat_weights, bins=rel_bin_edges)[0]

        sat_counts = np.array(count_binned, dtype='f8')
        smf_dist += sat_counts 

        coord_random_list, sat_bkg, flag_bkg = bkg(cat_neighbors_z_slice, coord_massive_gal, gal[mass_keyname])
        if flag_bkg == 0:  # success
            count_bkg += 1
            smf_dist_bkg += sat_bkg
            # print(gal['ID'],sum(sat_counts),sum(sat_bkg))

    # add results from this bootstrap iteration
    print('bkg/central counts', count_bkg, isolated_counts)
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

# output result to file
smf_dist_cen_tot = np.histogram(mass_centrals, bins=bin_edges)[0]
if len(mass_centrals) == isolated_counts and save_results:
    filename = path + 'smf_' + cat_name + '_cen_' + str(masscut_host) + '_' + str(r_low) + '_' + str(r_high) + '_' + str(masscut_low) + '_' + str(csfq) + '_' + str(ssfq) + '_' + str(round(z_min, 1))
    print(filename)

    print(rank_in_cat,'avg number total', ssfq, round(sum(smf_dist_avg)/isolated_counts))
    print(rank_in_cat,'avg number in bkg', ssfq, round(sum(smf_dist_bkg_avg)/isolated_counts))
    print(rank_in_cat,'avg number in sat', ssfq, round(sum(smf_dist_sat_avg)/isolated_counts))
    print(rank_in_cat,'isolated massive counts:', isolated_counts)

    # save total smf (not corrected for bkg)
    smf_dist_avg = np.append(smf_dist_avg, smf_dist_inf)
    smf_dist_avg = np.append(smf_dist_avg, smf_dist_sup)
    smf_dist_avg = np.append([isolated_counts], smf_dist_avg)
    np.save(filename + '_total_'+str(rank_in_cat), smf_dist_avg)

    # save bkg smf
    smf_dist_bkg_avg = np.append(smf_dist_bkg_avg, smf_dist_bkg_inf)
    smf_dist_bkg_avg = np.append(smf_dist_bkg_avg, smf_dist_bkg_sup)
    smf_dist_bkg_avg = np.append([isolated_counts], smf_dist_bkg_avg)
    np.save(filename + '_bkg_'+str(rank_in_cat), smf_dist_bkg_avg)

    # save sat smf
    smf_dist_sat_avg = np.append(smf_dist_sat_avg, smf_dist_sat_inf)
    smf_dist_sat_avg = np.append(smf_dist_sat_avg, smf_dist_sat_sup)
    smf_dist_sat_avg = np.append([isolated_counts], smf_dist_sat_avg)
    np.save(filename + '_sat_'+str(rank_in_cat), smf_dist_sat_avg)

    # save central gal list
    if ssfq == 'all':
        np.save(path + cat_name + '_' + str(r_low) + '_'+str(r_high) + '_' + str(masscut_low) + '_' + str(csfq) + '_' + str(round(z_min, 1)) + '_cen_mass_'+str(rank_in_cat), np.array(mass_centrals))

    if not rel_scale:
        np.save(path + 'bin_edges', bin_edges)
        # np.save(filename + '_cen', [smf_dist_cen_tot, isolated_counts])
    else:
        np.save(path + 'bin_edges', rel_bin_edges)

    
elif not save_results:
    print('isolated counts', isolated_counts, 'bkg counts', count_bkg)
else:
    print(len(mass_centrals), isolated_counts, 'Warning: wrong numbers! (Results not saved)')

