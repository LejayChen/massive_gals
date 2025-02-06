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


def bkg(cat_random_bkg_pos, cat_neighbors_z_slice_rand, coord_isolated_gal_rand, mass_cen, count_bkg):
    cat_neighbors_z_slice_rand = cat_neighbors_z_slice_rand[cat_neighbors_z_slice_rand[mass_keyname] > masscut_low]
    cat_neighbors_z_slice_rand = cat_neighbors_z_slice_rand[cat_neighbors_z_slice_rand[mass_keyname] < masscut_high]
    global z, overlapping_factor
    counts_gals_rand = np.zeros(bin_number)
    n = 0
    num_before_success = 0
    flag_bkg = 0

    if count_bkg == 0:
        max_iter = 100
    elif count_bkg <=3:
        max_iter = 20
    else:
        max_iter = 5

    while n < 1:  # get several blank pointing's to estimate background
        id_rand = int(random() * len(cat_random_bkg_pos))
        ra_rand = cat_random_bkg_pos[id_rand]['ra']
        dec_rand = cat_random_bkg_pos[id_rand]['dec']
        coord_rand = SkyCoord(ra_rand, dec_rand, unit="deg")
        idx, sep2d, dist3d = match_coordinates_sky(coord_rand, coord_isolated_gal_rand, nthneighbor=1)

        num_before_success += 1
        if num_before_success > max_iter:
            flag_bkg = 1
            break

        if sep2d.degree > overlapping_factor*2*r_iso/dis/np.pi*180:  # make sure the random pointing is away from any central galaxy (blank)
            if check_edge(ra_rand, dec_rand):
                continue

            if cat_name == 'ELAIS_deep':
                sph_coord_factor = 2
            else:
                sph_coord_factor = 1.1

            cat_neighbors_rand = cat_neighbors_z_slice_rand[np.logical_and(abs(cat_neighbors_z_slice_rand['RA'] - ra_rand) < r_iso*sph_coord_factor/dis/np.pi*180, abs(cat_neighbors_z_slice_rand['DEC'] - dec_rand) < r_iso/dis/np.pi*180 )]

            coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)
            if len(cat_neighbors_rand) == 0:
                continue

            # choose radial range
            if r_low>0.0:
                cat_neighbors_rand = cat_neighbors_rand[np.logical_and(coord_neighbors_rand.separation(coord_rand).degree < r_high/dis/np.pi*180, coord_neighbors_rand.separation(coord_rand).degree > r_low / dis / np.pi*180)]
            else:
                cat_neighbors_rand = cat_neighbors_rand[coord_neighbors_rand.separation(coord_rand).degree < r_high/dis/np.pi*180]

            if len(cat_neighbors_rand) == 0:
                continue
                counts_gals_rand += np.zeros(bin_number)
            elif max(cat_neighbors_rand[mass_keyname])>mass_cen:
                continue
            else:
                mass_neighbors_rand = cat_neighbors_rand[mass_keyname]

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

    # print(num_before_success)
    return counts_gals_rand, flag_bkg


# def correct_for_masked_area(ra, dec):
#     # correct area for normalization if it is partially in masked region
#     if not correct_masked:
#         return np.ones(bin_number), np.ones(bin_number)
#     else:
#         cat_nomask = cat_random_nomask[abs(cat_random_nomask['ra'] - ra) < r_iso / dis / np.pi * 180]
#         cat_nomask = cat_nomask[abs(cat_nomask['dec'] - dec) < r_iso / dis / np.pi * 180]
#         cat_nomask = cat_nomask[SkyCoord(cat_nomask['ra'] * u.deg, cat_nomask['dec'] * u.deg).separation
#                             (SkyCoord(ra * u.deg, dec * u.deg)).degree < r_iso / dis / np.pi * 180]

#         cat_mask = cat_random[abs(cat_random['ra'] - ra) < r_iso / dis / np.pi * 180]
#         cat_mask = cat_mask[abs(cat_mask['dec'] - dec) < r_iso / dis / np.pi * 180]
#         cat_mask = cat_mask[SkyCoord(cat_mask['ra'] * u.deg, cat_mask['dec'] * u.deg).separation
#                         (SkyCoord(ra * u.deg, dec * u.deg)).degree < r_iso / dis / np.pi * 180]

#         if len(cat_nomask) == 0:
#             return np.zeros(bin_number), np.zeros(bin_number)
#         else:
#             coord = SkyCoord(ra * u.deg, dec * u.deg)
#             coord_nomask = SkyCoord(cat_nomask['ra'] * u.deg, cat_nomask['dec'] * u.deg)
#             radius_list_nomask = coord_nomask.separation(coord).degree / 180. * np.pi * dis * 1000
#             count_nomask = np.histogram(radius_list_nomask, bins=bin_edges)[0]
#             count_nomask = np.array(count_nomask).astype(float)
#             if len(cat_mask) == 0:
#                 count_mask = np.zeros(bin_number)
#             else:
#                 coord_mask = SkyCoord(cat_mask['ra'] * u.deg, cat_mask['dec'] * u.deg)
#                 radius_list_mask = coord_mask.separation(coord).degree / 180. * np.pi * dis * 1000
#                 count_mask = np.histogram(radius_list_mask, bins=bin_edges)[0]
#                 count_mask = np.array(count_mask).astype(float)

#             return count_mask, count_nomask


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

def Ms_to_r200(log_Ms): 
    M1 = 10 ** 12.52 # solar masses
    Ms = 10**log_Ms # solar masses
    Ms0 = 10 ** 10.916 # solar masses
    beta = 0.457
    delta = 0.566
    gamma = 1.53
    rho_bar = 9.9e-30  # g/cm^3
    log_mh = np.log10(M1) + beta * np.log10(Ms / Ms0) + (Ms / Ms0) ** delta / (1 + (Ms / Ms0) ** (-1 * gamma)) - 0.5 # Leauthaud+2012
    r200 = ((3*10**log_mh*1.989e30*1e3)/(800*np.pi*rho_bar))**(1/3)/3.086e21 # in kpc
    return r200/1000  # in Mpc

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
small_size_test = False

ssfq = sys.argv[1]
z_min = eval(sys.argv[2])
z_max= eval(sys.argv[3])
path = sys.argv[4]
scratch_path = '/scratch/lejay/'+path 
bin_number = eval(sys.argv[5])
boot_num = int(eval(sys.argv[6]))

sigma_z = 0.06
masscut_low = eval(sys.argv[7])
masscut_high = 13.0
masscut_host = eval(sys.argv[8])
masscut_host_high = eval(sys.argv[9])
bin_edges = np.linspace(masscut_low, masscut_high, num=bin_number+1)
rel_bin_edges = np.linspace(-4, 0, num=bin_number+1)


sat_z_cut = 3.0
overlapping_factor = eval(sys.argv[10]) # closest distance factor a random position is to a massive galaxy (1=r_iso)
csfq = sys.argv[11]  # csf, cq, all

# radius of counting satellites
r_low = eval(sys.argv[12])  # Mpc
r_high = eval(sys.argv[13])  # Mpc
r_iso = r_high  # Mpc (isolation criteria radius)

overlap_exclude = bool(eval(sys.argv[14]))
rel_scale = bool(eval(sys.argv[15])) # relative mass scale
rel_riso = bool(eval(sys.argv[16])) # relative aperture size

print('relative scales status', rel_scale, rel_riso)

if not overlap_exclude:
    overlapping_factor = 0.0

if path[-1] != '/' and save_results:
    raise NameError('path is not a directory!')
elif save_results:
    print('will save results to ' + scratch_path)
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
cat_gal = cat_gal[cat_gal[mass_keyname] >= masscut_low]
cat_gal = cat_gal[cat_gal[mass_keyname] < masscut_high]

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

# trime cat_gal
cat_gal.remove_columns(['snr_i','snr_r','snr_z','i_cmodel','isStar','inside_hsc','isCompact','i_compact_flag'])

print(z_min,z_max)
print(cat_name,rank_in_cat,len(cat_gal))
del cat

#bootstrap resampling
smf_dist_arr = np.zeros(bin_number)
smf_dist_bkg_arr = np.zeros(bin_number)
mass_key_ori = cat_gal[mass_keyname].copy()  # original
z_key_ori = cat_gal[zkeyname].copy()  # original
mass_centrals_ori = []
isolated_counts_ori = 0
count_bkg_ori = 0
for boot_iter in range(boot_num):
    print('')
    print(cat_name, 'boot_iter '+str(boot_iter)+' started.')
    
    if boot_iter > 0:  # number of iteration of bootstrapping
        cat_gal[mass_keyname] = mass_key_ori
        cat_gal[zkeyname] = z_key_ori
        # z_scatter = np.random.normal(cat_gal[zkeyname], 0.06 * (cat_gal[zkeyname] + 1))  # photoz scatter
        # mass_scatter = np.log10(abs(np.random.normal(10 ** (cat_gal[mass_keyname] - 10),
        #                   (cat_gal['MASS_SUP'] - cat_gal['MASS_INF'])/2 * 10 ** (cat_gal['MASS_MED'] - 10)))) + 10  # mass scatter

        # cat_gal[mass_keyname] = mass_scatter
        # cat_gal[zkeyname] = z_scatter
        boot_idx = bootstrap(np.arange(len(cat_gal)), bootnum=1)
        cat_gal_copy = cat_gal[boot_idx[0].astype(int)]
    else:
        cat_gal_copy = cat_gal

    # select massive galaxies
    cat_massive_gal = cat_gal_copy[cat_gal_copy[mass_keyname] > masscut_host] # lower mass cut
    cat_massive_z_slice = cat_massive_gal[np.logical_and(cat_massive_gal[zkeyname]>z_min, cat_massive_gal[zkeyname]<z_max)]
    print('massive gals rank_in_cat:', rank_in_cat, len(cat_massive_z_slice), max(cat_massive_z_slice[mass_keyname]))

    #select isolated massive galaxies
    cat_random_non_proximity = cat_random_nomask.copy()
    isolated_ids = []
    for gal_idx, gal in enumerate(cat_massive_z_slice):  

        if rel_riso:
            r_high = Ms_to_r200(gal[mass_keyname])
            r_iso =  r_high

        dis = cosmo.angular_diameter_distance(gal[zkeyname]).value
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)

        cat_neighbors_massive = cat_massive_z_slice[np.logical_and(abs(cat_massive_z_slice['RA'] - gal['RA']) < 2 * r_iso / dis / np.pi * 180, abs(cat_massive_z_slice['DEC'] - gal['DEC']) < 2 * r_iso / dis / np.pi * 180)]

        # #### spatial selection
        coord_neighbors_massive= SkyCoord(cat_neighbors_massive['RA'] * u.deg, cat_neighbors_massive['DEC'] * u.deg)
        cat_neighbors_massive = cat_neighbors_massive[coord_neighbors_massive.separation(coord_gal).degree < r_iso / dis / np.pi * 180]

        # isolation cut on central
        if len(cat_neighbors_massive) == 0:  # central gals which has no companion
            continue
        elif gal[mass_keyname] < max(cat_neighbors_massive[mass_keyname]):  # no more-massive companions
            continue
        else:
            isolated_ids.append(gal_idx)
            # random point catalog to select backgroun positions
            cat_random_non_proximity = cat_random_non_proximity[np.logical_or(abs(cat_random_non_proximity['ra'] - gal['RA']) > r_iso / dis / np.pi * 180, abs(cat_random_non_proximity['dec'] - gal['DEC']) > r_iso / dis / np.pi * 180)]

    # isolated massive galaxy sample
    cat_isolated_z_slice = cat_massive_z_slice[isolated_ids]
    cat_isolated_z_slice = cat_isolated_z_slice[cat_isolated_z_slice[mass_keyname] < masscut_host_high] # upper mass cut
    coord_isolated_gal = SkyCoord(cat_isolated_z_slice['RA'] * u.deg, cat_isolated_z_slice['DEC'] * u.deg)
    print('isolated gals:', len(cat_isolated_z_slice), min(cat_isolated_z_slice[mass_keyname]), max(cat_isolated_z_slice[mass_keyname]))
    print('cat_random_non_proximity', len(cat_random_non_proximity))

    # write isolated galaxy catalog to file
    if (rank_in_cat==0 and boot_iter == 0) and ssfq == 'all':
        cat_isolated_z_slice.write(path + cat_name + '_cen_' + str(masscut_host) + '_' + str(round(z_min, 1))+'.fits', overwrite=True)

    # split massive gal sample 
    if rank_in_cat != cores_per_cat-1:
        cat_isolated_z_slice = cat_isolated_z_slice[rank_in_cat*int(len(cat_isolated_z_slice)/(nProcs/len(cat_names))) : (rank_in_cat+1)*int(len(cat_isolated_z_slice)/(nProcs/len(cat_names)))]
    else:
        cat_isolated_z_slice = cat_isolated_z_slice[rank_in_cat*int(len(cat_isolated_z_slice)/(nProcs/len(cat_names))) : (rank_in_cat+1)*int(len(cat_isolated_z_slice)/(nProcs/len(cat_names)))]

    # small scale test check
    if small_size_test:
        cat_isolated_z_slice_torun = cat_isolated_z_slice[np.random.choice(np.arange(len(cat_isolated_z_slice)), size=min(len(cat_isolated_z_slice),90), replace=False)]
    else:
        cat_isolated_z_slice_torun = cat_isolated_z_slice
    
    print('===rank:'+str(rank)+'===rank_in_cat=' + str(rank_in_cat) + '===z_min=' + str(round(z_min, 1)) + '===z_max=' + str(round(z_max, 1))+ '====='+cat_name+'==='+str(len(cat_isolated_z_slice))+'===')
    print('isolated gals, rank_in_cat:', rank_in_cat, len(cat_isolated_z_slice), min(cat_isolated_z_slice[mass_keyname]),max(cat_isolated_z_slice[mass_keyname]))

    # CORE CALCULATION
    isolated_counts = 0
    smf_dist = np.zeros(bin_number)
    smf_dist_bkg = np.zeros(bin_number)
    mass_centrals = []
    count_bkg = 0
    for gal in cat_isolated_z_slice_torun:  # [np.random.randint(len(cat_isolated_z_slice), size=300)]:

        if rel_riso:
            r_high = Ms_to_r200(gal[mass_keyname])
            r_iso =  r_high

        # cut on central SF/Q
        if csfq != 'all':
            if csfq == 'csf' and gal[sfq_keyname] < 0.5:
                continue
            elif csfq == 'cq' and gal[sfq_keyname] >= 0.5:
                continue

        dis = cosmo.angular_diameter_distance(gal[zkeyname]).value
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)

        # prepare neighbors catalog
        cat_neighbors_z_slice = cat_gal_copy[abs(cat_gal_copy[zkeyname] - gal[zkeyname]) < sat_z_cut * 0.06 * (1 + gal[zkeyname])]

        if cat_name == 'ELAIS_deep':
            sph_coord_factor = 2
        else:
            sph_coord_factor = 1.1

        cat_neighbors = cat_neighbors_z_slice[np.logical_and(abs(cat_neighbors_z_slice['RA'] - gal['RA']) < sph_coord_factor * r_iso / dis / np.pi * 180, abs(cat_neighbors_z_slice['DEC'] - gal['DEC']) < r_iso / dis / np.pi * 180 )]

        # choose sats within r_high
        coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
        cat_neighbors = cat_neighbors[coord_neighbors.separation(coord_gal).degree < r_high / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[cat_neighbors['ID'] != gal['ID']]
        if len(cat_neighbors) == 0:  # central gals which has no companion
            continue

        # exclude sats within r_low
        if r_low > 0.0:
            coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
            cat_neighbors = cat_neighbors[coord_neighbors.separation(coord_gal).degree >= r_low / dis / np.pi * 180]
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

        # sat_weights = np.array(sfq_weights/completeness_est(mass_neighbors, cat_neighbors[sfq_keyname], z))
        sat_weights = np.array(sfq_weights)

        #absolute / relative mass scale
        mass_neighbors = cat_neighbors[mass_keyname]
        if not rel_scale:
            count_binned = np.histogram(mass_neighbors, weights=sat_weights, bins=bin_edges)[0]
        else:
            rel_mass_neighbors = mass_neighbors - gal[mass_keyname]
            count_binned = np.histogram(rel_mass_neighbors, weights=sat_weights, bins=rel_bin_edges)[0]

        sat_counts = np.array(count_binned, dtype='f8')
        smf_dist += sat_counts 
        # print(gal['ID'],sum(sat_counts), gal[mass_keyname], max(cat_neighbors[mass_keyname]))

        if overlap_exclude:
            sat_bkg, flag_bkg = bkg(cat_random_non_proximity, cat_neighbors_z_slice, coord_isolated_gal, gal[mass_keyname], count_bkg)
        else:
            sat_bkg, flag_bkg = bkg(cat_random_nomask, cat_neighbors_z_slice, coord_isolated_gal, gal[mass_keyname], count_bkg)

        if flag_bkg == 0:  # success
            count_bkg += 1
            smf_dist_bkg += sat_bkg
            print(gal['ID'],sum(sat_counts), gal[mass_keyname],max(cat_neighbors[mass_keyname]), sum(sat_bkg))

    # add results from this bootstrap iteration
    print('bkg/central counts', count_bkg, isolated_counts, len(cat_isolated_z_slice))
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
    if not rel_riso:
        filename = scratch_path + 'smf_' + cat_name + '_cen_' + str(masscut_host) + '_' + str(r_low) + '_' + str(r_high) + '_' + str(masscut_low) + '_' + str(csfq) + '_' + str(ssfq) + '_' + str(round(z_min, 1))
    else:
        filename = scratch_path + 'smf_' + cat_name + '_cen_' + str(masscut_host) + '_rel_mscale_' + str(r_low) + '_r200_' + str(masscut_low) + '_' + str(csfq) + '_' + str(ssfq) + '_' + str(round(z_min, 1))
    print(filename)

    print(rank_in_cat,'avg number total', ssfq, round(sum(smf_dist_avg)/isolated_counts))
    print(rank_in_cat,'avg number in bkg', ssfq, round(sum(smf_dist_bkg_avg)/isolated_counts))
    print(rank_in_cat,'avg number in sat', ssfq, round(sum(smf_dist_sat_avg)/isolated_counts))
    print(rank_in_cat,'isolated massive counts:', isolated_counts)

    # print errors
    print(smf_dist_avg)
    print(np.sqrt(smf_dist_avg))
    print(smf_dist_avg - smf_dist_inf)

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
    # if ssfq == 'all':
    #     np.save(scratch_path + cat_name + '_' + str(r_low) + '_'+str(r_high) + '_' + str(masscut_low) + '_' + str(csfq) + '_' + str(round(z_min, 1)) + '_cen_mass_'+str(rank_in_cat), np.array(mass_centrals))

    if not rel_scale:
        np.save(scratch_path + 'bin_edges', bin_edges)
        # np.save(filename + '_cen', [smf_dist_cen_tot, isolated_counts])
    else:
        np.save(scratch_path + 'rel_bin_edges', rel_bin_edges)

    
elif not save_results:
    print('isolated counts', isolated_counts, 'bkg counts', count_bkg)
else:
    print(len(mass_centrals), max(mass_centrals), isolated_counts, 'Warning: wrong numbers! (Results not saved)')

