import sys
from random import random
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.cosmology import WMAP9
from astropy.table import Table
from scipy import stats
from sklearn.neighbors import KDTree
from plot_bkg import *
import matplotlib.pyplot as plt
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

sys.path.append('../')
from radial_functions import *


def bkg(cat_neighbors_z_slice_rand, coord_massive_gal_rand, mode='count'):
    '''
    :return
        coord_rand_list:
           a list of coordinates of random point selected as where random apertures are placed
        count_gals_rand:
            an array of companion count per central per bin'''
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
        if num_before_success > 25:
            flag_bkg = 1
            break
        if sep2d.degree > 0.3/dis/np.pi*180:  # make sure the random pointing is away from any central galaxy (blank)
            num_before_success = 0
            coord_rand = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)
            coord_rand_list.append(coord_rand)
            cat_neighbors_rand = cat_neighbors_z_slice_rand[abs(cat_neighbors_z_slice_rand['RA'] - ra_rand) < 0.7/dis/np.pi*180]
            cat_neighbors_rand = cat_neighbors_rand[abs(cat_neighbors_rand['DEC'] - dec_rand) < 0.7/dis/np.pi*180]
            coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)
            cat_neighbors_rand = cat_neighbors_rand[coord_neighbors_rand.separation(coord_rand).degree < 0.7/dis/np.pi*180]
            
            # make some cuts
            cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand[mass_keyname] > masscut_low]
            cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand[mass_keyname] < masscut_high]
            if ssfq == 'ssf':
                cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['SSFR_BEST'] > -11]
            elif ssfq == 'sq':
                cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['SSFR_BEST'] < -11]

            coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)
            radius_neighbors_rand = coord_neighbors_rand.separation(coord_rand).degree/180.*np.pi*dis*1000
            if len(radius_neighbors_rand) != 0:
                mass_neighbors_rand = cat_neighbors_rand[mass_keyname]
                ssfr_neighbors_rand = cat_neighbors_rand['SSFR_BEST']
                if mode == 'count':
                    # count_gal_rand, edges_rand = np.histogram(radius_neighbors_rand, bins=bin_edges)
                    # counts_gals_rand += count_gal_rand / correct_for_masked_area(ra_rand, dec_rand)
                    counts_gals_rand += sum(1./completeness_est(mass_neighbors_rand, ssfr_neighbors_rand, z)) * areas / (np.pi*bin_edges[-1]**2)  # smoothed background, assuming bkg is uniform
                    R_m, R_u = correct_for_masked_area(ra_rand, dec_rand)
                    R_m_stack_bkg += R_m
                    R_u_stack_bkg += R_u
                elif mode == 'mass':
                    binned_data_rand = stats.binned_statistic(radius_neighbors_rand, 10**(mass_neighbors_rand - 10), statistic='sum', bins=bin_edges)
                    mass_binned_rand = binned_data_rand[0]
                    counts_gals_rand += mass_binned_rand
            else:
                counts_gals_rand += np.zeros(bin_number)
            n = n + 1

    return coord_rand_list, counts_gals_rand/float(num_p), flag_bkg


def cut_random_cat(cat_rand, coord_list):
    # cut random point catalog to avoid overlapping
    for coord in coord_list:
        coord_rand_list = SkyCoord(cat_rand['RA'] * u.deg, cat_rand['DEC'] * u.deg)
        cat_rand = cat_rand[coord_rand_list.separation(coord).degree > 1.05 / dis / np.pi * 180]
    add_to_random_points(coord_list)
    return cat_rand


def add_to_random_points(coord_list):
    # store coord of selected random points just for record
    for coord in coord_list:
        cat_random_points.add_row([coord.ra.value, coord.dec.value, gal['NUMBER']])
    return 0


def correct_for_masked_area(ra, dec):
    # correct area for normalization if it is partially in masked region

    cat_nomask = cat_random_nomask[abs(cat_random_nomask['RA'] - ra) < 0.7 / dis / np.pi * 180]
    cat_nomask = cat_nomask[abs(cat_nomask['DEC'] - dec) < 0.7 / dis / np.pi * 180]
    cat_nomask = cat_nomask[SkyCoord(cat_nomask['RA']*u.deg, cat_nomask['DEC']*u.deg).separation
                            (SkyCoord(ra*u.deg, dec*u.deg)).degree < 0.7/dis/np.pi*180]

    cat_mask = cat_random[abs(cat_random['RA'] - ra) < 0.7 / dis / np.pi * 180]
    cat_mask = cat_mask[abs(cat_mask['DEC'] - dec) < 0.7 / dis / np.pi * 180]
    cat_mask = cat_mask[SkyCoord(cat_mask['RA'] * u.deg, cat_mask['DEC'] * u.deg).separation
                            (SkyCoord(ra * u.deg, dec * u.deg)).degree < 0.7 / dis / np.pi * 180]

    if len(cat_mask) == 0:  # point outside random point catalog region
        return np.ones(bin_number), np.ones(bin_number)
    else:
        coord = SkyCoord(ra * u.deg, dec * u.deg)
        coord_nomask = SkyCoord(cat_nomask['RA'] * u.deg, cat_nomask['DEC'] * u.deg)
        coord_mask = SkyCoord(cat_mask['RA'] * u.deg, cat_mask['DEC'] * u.deg)
        radius_list_nomask = coord_nomask.separation(coord).degree / 180. * np.pi * dis * 1000
        radius_list_mask = coord_mask.separation(coord).degree / 180. * np.pi * dis * 1000
        count_nomask = np.histogram(radius_list_nomask, bins=bin_edges)[0]
        count_mask = np.histogram(radius_list_mask, bins=bin_edges)[0]
        count_nomask = np.array(count_nomask).astype(float)
        count_mask = np.array(count_mask).astype(float)
        
        if len(count_mask[count_mask != 0]) < 3 and len(count_nomask[count_nomask != 0]) > 2:  # point on the edge of random point catalog region
            return np.ones(bin_number), np.ones(bin_number)

        count_mask[count_mask==0.] = 1
        count_nomask[count_nomask==0.] = 1
        return count_mask, count_nomask


def completeness_est(mass_list, ssfr_list, z):
    try:
        completeness_sf = np.genfromtxt('../mass_completeness_data/allFields_' + str(round(z - 0.1, 1)) + '_z_' + str(round(z + 0.1,1)) + '_sf_nopert_nan.txt')
        completeness_q = np.genfromtxt('../mass_completeness_data/allFields_' + str(round(z - 0.1, 1)) + '_z_' + str(round(z + 0.1,1)) + '_q_nopert_nan.txt')
        completeness = np.array([])
        for idx in range(len(mass_list)):
            if ssfr_list[idx] > -11:
                completeness = np.append(completeness,np.interp(mass_list[idx], completeness_sf[0], completeness_sf[3]))
            else:
                completeness = np.append(completeness,np.interp(mass_list[idx], completeness_q[0], completeness_q[3]))

        completeness[np.isnan(completeness)] = 1.
        return completeness
    except:
        return np.ones(len(mass_list))


def spatial_comp():
    comp_sf = 1
    comp_q = 1
    comp_all = 1
    return 1

# ################# START #####################
cat_name = sys.argv[1]  # COSMOS_deep COSMOS_uddd ELAIS_deep XMM-LSS_deep DEEP_deep SXDS_uddd
mode = sys.argv[2]  # 'count' or 'mass'
z_keyname = 'zKDEPeak'
mass_keyname = 'MASS_MED'
masscut_h_low = 10.8
masscut_h_high = 11.0
masscut_low = 9.5
masscut_high = np.inf
csfq = 'all'  # csf, cq, all
ssfq = 'all'  # ssf, sq, all
path = 'split_host_mass/'
bin_number = 14
bin_edges = 10 ** np.linspace(1, 2.845, num=bin_number+1)
areas = np.array([])
for i in range(len(bin_edges[:-1])):
    areas = np.append(areas, (bin_edges[i + 1] ** 2 - bin_edges[i] ** 2) * np.pi)
print(mode, csfq, ssfq, masscut_low, masscut_high)
print('start reading catalogs', end='\r')

# read in data catalog
cat = Table.read('CUT2_'+cat_name+'.fits')
cat = cat[cat[z_keyname] < 1]
cat_gal = cat[np.logical_and(cat['preds_median'] < 0.89, cat['inside'] == True)]
cat_gal = cat_gal[cat_gal[mass_keyname] > masscut_low]
cat_massive_gal = cat_gal[cat_gal[mass_keyname] > masscut_h_low]
cat_massive_gal = cat_massive_gal[cat_massive_gal[mass_keyname] < masscut_h_high]

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
for z in np.arange(6, 6.1, 1)/10.:
    z_bin_size = 0.1
    print('============='+str(round(z, 1))+'================')
    cat_random_copy = np.copy(cat_random)  # reset random points catalog at each redshift
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal[z_keyname] - z) < z_bin_size]
    # cat_massive_z_slice = cat_massive_z_slice[np.random.rand(len(cat_massive_z_slice)) < 0.9]
    coord_massive_gal = SkyCoord(cat_massive_z_slice['RA'] * u.deg, cat_massive_z_slice['DEC'] * u.deg)
    cat_all_z_slice = cat_gal[abs(cat_gal[z_keyname] - z) < 0.5]
    cat_all_z_slice_bkg = cat_gal[abs(cat_gal[z_keyname] - z) < 1.5 * 0.044 * (1 + z)]

    radial_dist = 0
    radial_dist_bkg = 0
    radial_count_dist = 0
    isolated_counts = len(cat_massive_z_slice)
    massive_count = 0
    bin_centers_stack = 0
    companion_count_stack = 0
    R_m_stack = np.ones(bin_number)*1e-6  # number of random points (masked) in each bin
    R_u_stack = np.ones(bin_number)*1e-6  # number of random points (unmasked) in each bin
    R_m_stack_bkg = np.ones(bin_number)*1e-6  # number of random points (masked, bkg apertures) in each bin
    R_u_stack_bkg = np.ones(bin_number)*1e-6  # number of random points (unmasked, bkg apertures) in each bin
    for gal in cat_massive_z_slice:
        massive_count += 1
        print('Progress:'+str(massive_count)+'/'+str(len(cat_massive_z_slice)), end='\r')
        dis = WMAP9.angular_diameter_distance(gal[z_keyname]).value
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)

        # prepare neighbors catalog
        cat_neighbors_z_slice = cat_all_z_slice[abs(cat_all_z_slice[z_keyname] - gal[z_keyname]) < 1.5 * 0.044 * (1 + z)]
        cat_neighbors = cat_neighbors_z_slice[abs(cat_neighbors_z_slice['RA'] - gal['RA']) < 0.7 / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[abs(cat_neighbors['DEC'] - gal['DEC']) < 0.7 / dis / np.pi * 180]

        if len(cat_neighbors) == 0:  # central gals which has no companion
            continue
        else:
            ind = KDTree(np.array(cat_neighbors['RA', 'DEC']).tolist()).query_radius([(gal['RA'], gal['DEC'])], 0.7 / dis / np.pi * 180)
            cat_neighbors = cat_neighbors[ind[0]]
            cat_neighbors = cat_neighbors[cat_neighbors['NUMBER'] != gal['NUMBER']]
            if len(cat_neighbors) == 0:  # central gals which has no companion
                continue

        # isolation cut on central
        if gal[mass_keyname] < max(cat_neighbors[mass_keyname]):  # no more-massive companions
            isolated_counts -= 1
            continue

        # cut on central SF/Q
        if csfq == 'csf' and gal['SSFR_BEST'] < -11:
            isolated_counts -= 1
            continue
        elif csfq == 'cq' and gal['SSFR_BEST'] > -11:
            isolated_counts -= 1
            continue

        # cut on companion sample (cut the final sample)
        cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] > masscut_low]
        cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] < masscut_high]
        if ssfq == 'ssf':
            cat_neighbors = cat_neighbors[cat_neighbors['SSFR_BEST'] > -11]
        elif ssfq == 'sq':
            cat_neighbors = cat_neighbors[cat_neighbors['SSFR_BEST'] < -11]
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
                bin_centers[i] = (bin_edges[i]+bin_edges[i+1])/2
        bin_centers_stack += bin_centers*len(radius_list)
        companion_count_stack += len(radius_list)

        # Core Function: statistics #
        count_binned = np.histogram(radius_list, weights=np.array(1./completeness_est(mass_neighbors, ssfr_neighbors,z)), bins=bin_edges)[0]
        sat_counts = np.array(count_binned, dtype='f8')
        R_m, R_u = correct_for_masked_area(gal['RA'], gal['DEC'])
        R_m_stack += R_m
        R_u_stack += R_u
        if mode == 'count':                                             # radial distribution histogram
            radial_dist += sat_counts
        else:                                                           # mass distribution histogram
            binned_data = stats.binned_statistic(radius_list, 10**(mass_neighbors-10), statistic='sum', bins=bin_edges)
            sat_masses = np.array(binned_data[0], dtype='f8')
            radial_dist += sat_masses
            radial_count_dist += sat_counts

    # background estimation
    flag_bkg = 0
    count_bkg = 0
    while flag_bkg == 0 and count_bkg < isolated_counts:
        print('Bkg Progress:' + str(count_bkg) + '                                       ', end='\r')
        coord_random_list, sat_bkg, flag_bkg = bkg(cat_all_z_slice_bkg, coord_massive_gal, mode=mode)
        if flag_bkg == 0:
            radial_dist_bkg += sat_bkg
            cat_random_copy = cut_random_cat(cat_random_copy, coord_random_list)
            count_bkg += 1

    #aggregate results
    radial_dist_norm = radial_dist*R_u_stack/R_m_stack/float(isolated_counts)/areas
    radial_dist_bkg_norm = radial_dist_bkg*R_u_stack_bkg/R_m_stack_bkg/float(count_bkg)/areas
    if mode == 'count':
        err = cal_error2(radial_dist, radial_dist_bkg, isolated_counts) / areas
    else:
        err = cal_error2(radial_dist, radial_dist_bkg, isolated_counts)*(radial_dist / radial_count_dist) / areas
    n_central, n_count, n_err = isolated_counts, radial_dist_norm - radial_dist_bkg_norm, err
    result = np.append([n_central], [n_count, n_err])

    # output result to file
    filename = path + str(mode) + cat_name + '_' +'host_' +str(masscut_h_low)+'_'+str(masscut_low) + '_' + str(csfq) + '_' + str(ssfq) + '_' + str(round(z, 1))
    if sum(R_u_stack) < 1:
        # no massive galaxies have desired satellite population
        np.savetxt(filename,result)
        print('No massive galaxy with desired satellites!')
    else:
        np.savetxt(filename,result)
        np.save(path + 'bin_edges', bin_edges)
        np.save(path + 'bin_centers', bin_centers_stack / companion_count_stack)

    # cat_random_points.write('random_points_'+cat_name+'_'+str(round(z,1))+'.fits', overwrite=True)
    # plot_bkg(cat_name,z)
    print('massive counts:', len(cat_massive_z_slice), isolated_counts)
