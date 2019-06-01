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

def bkg(cat_neighbors_z_slice_rand, coord_massive_gal_rand, m_cen):

    global R_m_stack_bkg, R_u_stack_bkg, z, cat_random_copy
    coord_rand_list = []
    counts_gals_rand = 0
    num_before_success = 0
    flag_bkg = 1

    while flag_bkg == 1:
        id_rand = int(random() * len(cat_random_copy))
        ra_rand = cat_random_copy[id_rand]['RA']
        dec_rand = cat_random_copy[id_rand]['DEC']
        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(ra_rand, dec_rand, unit="deg"), coord_massive_gal_rand, nthneighbor=1)
        if sep2d.degree > 1.6 / dis / np.pi * 180:  # make sure the random pointing is away from any central galaxy (blank)
            coord_rand = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)
            coord_rand_list.append(coord_rand)
            cat_random_copy = cut_random_cat(cat_random_copy, coord_rand_list)
            flag_bkg = 0

        num_before_success += 1
        if num_before_success > 30:
            return 0, 0, 0, flag_bkg

    # "neighbors" selection
    cat_neighbors_rand = cat_neighbors_z_slice_rand[abs(cat_neighbors_z_slice_rand['RA'] - ra_rand) < 0.7 / dis / np.pi * 180]
    cat_neighbors_rand = cat_neighbors_rand[abs(cat_neighbors_rand['DEC'] - dec_rand) < 0.7 / dis / np.pi * 180]
    coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)
    cat_neighbors_rand = cat_neighbors_rand[coord_neighbors_rand.separation(coord_rand).degree < 0.7 / dis / np.pi * 180]

    # make some mass cuts
    cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand[mass_keyname] > masscut_low]
    cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand[mass_keyname] < masscut_high]

    # statistics: count total/merging "satellites"
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

        counts_gals_rand = np.histogram(radius_neighbors_rand, weights=np.array(sfq_weights), bins=bin_edges)[0]  # smoothed background, assuming bkg is uniform
        counts_gals_rand_merge, mass_rand_merge = n_merge(z, z-0.2, radius_neighbors_rand, mass_neighbors_rand ,gal['MASS_MED'])
            
        R_m, R_u = correct_for_masked_area(ra_rand, dec_rand)
        R_m_stack_bkg += R_m
        R_u_stack_bkg += R_u
    else:
        counts_gals_rand = np.zeros(bin_number)
        counts_gals_rand_merge = 0
        mass_rand_merge = 0

    return counts_gals_rand ,counts_gals_rand_merge, mass_rand_merge, flag_bkg


def cut_random_cat(cat_rand, coord_list):
    # cut random point catalog to avoid overlapping
    for coord in coord_list:
        coord_rand_list = SkyCoord(cat_rand['RA'] * u.deg, cat_rand['DEC'] * u.deg)
        cat_rand = cat_rand[coord_rand_list.separation(coord).degree > 1.4 / dis / np.pi * 180]
    return cat_rand

def correct_for_masked_area(ra, dec):
    # correct area for normalization if it is partially in masked region

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
    return spatial_weight, spatial_weight_err

    
def cal_error(n_sample, n_bkg, n_central, w_comp, w_comp_err):
    n_comp = np.array(n_sample - n_bkg).astype(float)
    n_comp[n_comp==0] = 1

    sigma_n_sample_ori = np.sqrt(n_sample/w_comp)
    sigma_n_sample_corr = np.sqrt(sigma_n_sample_ori**2*w_comp**2 + (n_sample/w_comp)**2*w_comp_err**2)
    sigma_n_bkg = np.sqrt(n_bkg)
    sigma_n_comp = np.sqrt(sigma_n_sample_corr**2+sigma_n_bkg**2)
    sigma_n_central = np.array(np.sqrt(n_central)).astype(float)

    errors = (n_comp/n_central)*np.sqrt((sigma_n_comp/n_comp)**2+(sigma_n_central/n_central)**2)
    return errors


def merge_est_k(r, m_cen, z):
    m = 10 ** (m_cen - 10)
    h = 1
    t_merge = 3.2*(r/50.) * (m/4*h)**(-0.3) * (1+z/20.)  # Gyr

    return np.array(t_merge)


def merge_est_j(r, m_sat, m_cen):
    m_sat = 10 ** (m_sat - 10)
    m_cen = 10 ** (m_cen - 10)
    r = r*3.086e16  # kpc to km
    t_merge = (0.94*0.5**0.6+0.6)/0.86 * (m_cen/m_sat) * (1/np.log(1+m_cen/m_sat)) * r/860.  # second
    t_merge = t_merge/(365.25*24*60*60)/1e9  # second to Gyr

    return np.array(t_merge)


def n_merge(z1, z2, r_list, m_sat, m_cen):
    t_between = WMAP9.lookback_time(z1).value - WMAP9.lookback_time(z2).value
    # t_merge_list = merge_est_k(r_list, m_cen, z1)
    t_merge_list = merge_est_j(r_list, m_sat, m_cen)

    # merge
    m_sat = m_sat[t_merge_list<t_between]
    m_merge = sum(10**(m_sat-8))
    n_merge = sum(np.ones(len(r_list))*t_merge_list/t_between)

    # alternatively
    # n_merge = r_list[t_merge_list<t_between]

    return n_merge, m_merge


# ################# START #####################
cat_name = sys.argv[1]  # COSMOS_deep COSMOS_uddd ELAIS_deep XMM-LSS_deep DEEP_deep SXDS_uddd
z_keyname = 'zKDEPeak'
mass_keyname = 'MASS_MED'
masscut_low = 9.5
masscut_high = 13.0
path = 'number_counts/'
csfq = 'all'  # csf, cq, all
ssfq = sys.argv[2]  # ssf, sq, all
ssfr_cut = -10.5
sfprob_cut = 0.5

# setting binning scheme
bin_number = 14
bin_edges = 10 ** np.linspace(1.0, 2.845, num=bin_number + 1)
areas = np.array([])
for i in range(len(bin_edges[:-1])):
    areas = np.append(areas, (bin_edges[i + 1] ** 2 - bin_edges[i] ** 2) * np.pi)

# read in data catalog
print('start reading catalogs', end='\r ')
cat = Table(fits.getdata('CUT3_' + cat_name + '.fits'))
cat = cat[cat[z_keyname] < 1]
cat_gal = cat[np.logical_and(cat['preds_median'] < 0.89, cat['inside'] == True)]
cat_gal = cat_gal[cat_gal[mass_keyname] > masscut_low]

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
for z in [0.4, 0.6, 0.8]:
    z = round(z, 1)
    z_bin_size = 0.1
    masscut_low_host = 11.15
    masscut_high_host = 13.0
    print(csfq, ssfq, masscut_low, masscut_high, masscut_low_host)
    print('=============' + str(round(z, 1)) + '================')
    cat_random_copy = np.copy(cat_random)  # reset random points catalog at each redshift

    cat_massive_gal = cat_gal[np.logical_and(cat_gal[mass_keyname] > masscut_low_host, cat_gal[mass_keyname] < masscut_high_host)]
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal[z_keyname] - z) < z_bin_size]
    coord_massive_gal = SkyCoord(cat_massive_z_slice['RA'] * u.deg, cat_massive_z_slice['DEC'] * u.deg)
    cat_all_z_slice = cat_gal[abs(cat_gal[z_keyname] - z) < 0.5]
    cat_all_z_slice_bkg = cat_gal[abs(cat_gal[z_keyname] - z) < 1.5 * 0.044 * (1 + z)]

    # initiations
    isolated_counts = len(cat_massive_z_slice)
    csf_counts = 0
    cq_counts = 0
    massive_count = 0
    bkg_count = 0
    R_m_stack = np.ones(bin_number) * 1e-6  # number of random points (masked) in each bin
    R_u_stack = np.ones(bin_number) * 1e-6  # number of random points (unmasked) in each bin
    R_m_stack_bkg = np.ones(bin_number) * 1e-6  # number of random points (masked, bkg apertures) in each bin
    R_u_stack_bkg = np.ones(bin_number) * 1e-6  # number of random points (unmasked, bkg apertures) in each bin
    
    n_sat_collect = []
    n_sat_merge = []
    mass_sat_merge = []
    n_sat_collect_csf = []
    n_sat_collect_cq = []

    for gal in cat_massive_z_slice:
        massive_count += 1
        print('Progress:' + str(massive_count) + '/' + str(len(cat_massive_z_slice)), end='\r')
        dis = WMAP9.angular_diameter_distance(gal[z_keyname]).value
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)

        # prepare neighbors catalog
        cat_neighbors_z_slice = cat_all_z_slice[abs(cat_all_z_slice[z_keyname] - gal[z_keyname]) < 1.5 * 0.044 * (1 + z)]
        cat_neighbors = cat_neighbors_z_slice[abs(cat_neighbors_z_slice['RA'] - gal['RA']) < 0.7 / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[abs(cat_neighbors['DEC'] - gal['DEC']) < 0.7 / dis / np.pi * 180]

        if len(cat_neighbors) == 0:  # central gals which has no companion
            print(gal['NUMBER'])
            continue
        else:
            ind = KDTree(np.array(cat_neighbors['RA', 'DEC']).tolist()).query_radius([(gal['RA'], gal['DEC'])],0.7 / dis / np.pi * 180)
            cat_neighbors = cat_neighbors[ind[0]]
            cat_neighbors = cat_neighbors[cat_neighbors['NUMBER'] != gal['NUMBER']]
            if len(cat_neighbors) == 0:  # central gals which has no companion
                print(gal['NUMBER'])
                continue

        # isolation cut on central
        if gal[mass_keyname] < max(cat_neighbors[mass_keyname]):  # no more-massive companions
            isolated_counts -= 1
            continue

        # cut on companion sample (cut the final sample)
        cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] > masscut_low]
        cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] < masscut_high]
        if len(cat_neighbors) == 0:  # central gals which has no companion
            continue

        # list of satellite properties (mass,ssfr, radius)
        mass_neighbors = cat_neighbors[mass_keyname]
        ssfr_neighbors = cat_neighbors['SSFR_BEST']
        coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
        radius_list = coord_neighbors.separation(coord_gal).degree / 180. * np.pi * dis * 1000  # kpc

        # Core Function: total/merging satellite number counts #
        if ssfq == 'all':
            sfq_weights = np.ones(len(radius_list))
        elif ssfq == 'ssf':
            sfq_weights = cat_neighbors['sfProb']
        else:
            sfq_weights = 1 - cat_neighbors['sfProb']

        count_binned = np.histogram(radius_list, weights=np.array(sfq_weights), bins=bin_edges)[0]
        sat_counts = np.array(count_binned, dtype='f8')
        sat_counts_merge, sat_mass_merge = n_merge(z, z-0.2, radius_list, mass_neighbors, gal['MASS_MED'])
        R_m, R_u = correct_for_masked_area(gal['RA'], gal['DEC'])
        R_m_stack += R_m
        R_u_stack += R_u

        sat_counts_bkg, sat_counts_merge_bkg, sat_mass_merge_bkg, flag_bkg = bkg(cat_neighbors_z_slice, coord_massive_gal, gal['MASS_MED'])

        num_sat_norm = sat_counts - sat_counts_bkg
        num_merge = sat_counts_merge - sat_counts_merge_bkg
        mass_merge = sat_mass_merge - sat_mass_merge_bkg

        # collect results
        n_sat_collect.append(sum(num_sat_norm))
        n_sat_merge.append(num_merge)
        mass_sat_merge.append(mass_merge)
        if gal['sfProb'] < sfprob_cut:
            n_sat_collect_cq.append(sum(num_sat_norm))
            cq_counts += 1
        elif gal['sfProb'] > sfprob_cut:
            n_sat_collect_csf.append(sum(num_sat_norm))
            csf_counts += 1

    # completeness correction
    spatial_weight, spatial_weight_err = spatial_comp(z, masscut_low, masscut_high, ssfq)
    spatial_weight = np.average(spatial_weight, weights=areas)
    n_sat_collect = np.array(n_sat_collect) * float(spatial_weight)
    n_sat_merge = np.array(n_sat_merge) * float(spatial_weight)
    n_sat_collect_csf = np.array(n_sat_collect_csf) * float(spatial_weight)
    n_sat_collect_cq = np.array(n_sat_collect_cq) * float(spatial_weight)
    mass_sat_merge = np.array(mass_sat_merge) * float(spatial_weight)

    # mask correction
    mask_weight = np.average(R_u_stack / R_m_stack, weights=areas)
    n_sat_collect = np.array(n_sat_collect) * float(mask_weight)
    n_sat_collect_csf = np.array(n_sat_collect_csf) * float(mask_weight)
    n_sat_collect_cq = np.array(n_sat_collect_cq) * float(mask_weight)

    # output result to file
    print('Finished gals: '+str(isolated_counts)+'/'+str(len(cat_massive_z_slice)))
    print('Finished bkgs: '+str(bkg_count))
    result = [isolated_counts,csf_counts, cq_counts,np.average(n_sat_collect), np.average(n_sat_merge),np.average(n_sat_collect_csf), np.average(n_sat_collect_cq)]
    print(result)
    
    np.savetxt(path+'number_'+cat_name+'_'+str(round(z,1))+'.txt',result)
    np.savetxt(path+'number_sathist_'+cat_name+'_'+str(round(z,1))+'.txt',n_sat_collect)
    np.savetxt(path+'number_sathist_merge_'+cat_name+'_'+str(round(z,1))+'.txt',n_sat_merge)
    np.savetxt(path+'number_sathist_csf_'+cat_name+'_'+str(round(z,1))+'.txt',n_sat_collect_csf)
    np.savetxt(path+'number_sathist_cq_'+cat_name+'_'+str(round(z,1))+'.txt',n_sat_collect_cq)
    np.savetxt(path+'mass_sathist_'+cat_name+'_'+str(round(z,1))+'.txt',mass_sat_merge)