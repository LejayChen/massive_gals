import sys, os
from random import random
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.cosmology import WMAP9
from astropy.table import *
# from scipy.stats import binned_statistic
from mpi4py import MPI
import time
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
nProcs = comm.Get_size()


def check_edge(ra_rand, dec_rand, dis):
    cat_random_cut = cat_random_copy[abs(cat_random_copy[ra_key] - ra_rand) < radius_max*4 / dis / np.pi * 180]
    cat_random_cut = cat_random_cut[abs(cat_random_cut[dec_key] - dec_rand) < radius_max*4 / dis / np.pi * 180]
    coord_random_cut = SkyCoord(np.array(cat_random_cut[ra_key]) * u.deg, np.array(cat_random_cut[dec_key]) * u.deg)
    cat_random_cut = cat_random_cut[coord_random_cut.separation(coord_random_cut).degree < radius_max / dis / np.pi * 180]
    try:
        ra_ratio = len(cat_random_cut[cat_random_cut[ra_key] < ra_rand])/len(cat_random_cut[cat_random_cut[ra_key] > ra_rand])
        dec_ratio = len(cat_random_cut[cat_random_cut[dec_key] < dec_rand])/len(cat_random_cut[cat_random_cut[dec_key] > dec_rand])
    except (ValueError, ZeroDivisionError):
        return True

    if ra_ratio > 1/0.75 or ra_ratio < 0.75:
        return True
    elif dec_ratio > 1/0.75 or dec_ratio < 0.75:
        return True
    else:
        return False


def bkg(cat_neighbors_z_slice_rand, coord_massive_gal_rand, mass_cen, dis):

    cat_neighbors_z_slice_rand = cat_neighbors_z_slice_rand[cat_neighbors_z_slice_rand[mass_keyname] > masscut_low]
    cat_neighbors_z_slice_rand = cat_neighbors_z_slice_rand[cat_neighbors_z_slice_rand[mass_keyname] < masscut_high]

    global R_m_stack_bkg, R_u_stack_bkg, z
    flag_bkg_rand = -1
    num_before_success = 0
    counts_gals_rand = np.zeros(bin_number)
    counts_gals_ssf_rand = np.zeros(bin_number)
    counts_gals_sq_rand = np.zeros(bin_number)
    coord_rand_list = []
    while flag_bkg_rand == -1:  # get several blank pointing's to estimate background
        id_rand = int(random() * len(cat_random_copy))
        ra_rand = cat_random_copy[id_rand][ra_key]
        dec_rand = cat_random_copy[id_rand][dec_key]
        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(ra_rand, dec_rand, unit="deg"), coord_massive_gal_rand, nthneighbor=1)

        num_before_success += 1
        if num_before_success > 100:
            flag_bkg_rand = 1
            break

        if sep2d.degree > radius_max*2 / dis / np.pi * 180:  # make sure the random pointing is away from any central galaxy (blank)
            if check_edge_flag and check_edge(ra_rand, dec_rand, dis):
                continue

            coord_rand = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)
            cat_neighbors_rand = cat_neighbors_z_slice_rand[abs(cat_neighbors_z_slice_rand[ra_key] - ra_rand) < radius_max * 4 / dis / np.pi * 180]
            cat_neighbors_rand = cat_neighbors_rand[abs(cat_neighbors_rand[dec_key] - dec_rand) < radius_max*4 / dis / np.pi * 180]
            coord_neighbors_rand = SkyCoord(np.array(cat_neighbors_rand[ra_key]) * u.deg, np.array(cat_neighbors_rand[dec_key]) * u.deg)
            cat_neighbors_rand = cat_neighbors_rand[coord_neighbors_rand.separation(coord_rand).degree < radius_max / dis / np.pi * 180]

            # exclude bkg apertures that contains galaxies more massive than central
            # if len(cat_neighbors_rand) > 0:
            #     if max(cat_neighbors_rand[mass_keyname]) < mass_cen:
            #         num_before_success = 0
            #     else:
            #         continue
            # else:
            #     continue

            # create radius list
            coord_neighbors_rand = SkyCoord(np.array(cat_neighbors_rand[ra_key]) * u.deg, np.array(cat_neighbors_rand[dec_key]) * u.deg)
            radius_neighbors_rand = coord_neighbors_rand.separation(coord_rand).degree / 180. * np.pi * dis * 1000

            # save bkg obj catalog
            if ('all' in ssfq_series) and save_catalogs:
                radius_col_rand = Column(data=radius_neighbors_rand, name='radius', dtype='i4')
                cat_neighbors_rand.add_columns([radius_col_rand])
                cat_neighbors_rand.write(sat_cat_dir + cat_name + '_' + str(gal[id_keyname]) + '_bkg.fits', overwrite=True)

            # radial distribution of bkg objects
            if len(radius_neighbors_rand) != 0:
                sfq_weights_rand = np.ones(len(radius_neighbors_rand))
                sfq_weights_ssf_rand = cat_neighbors_rand[sfq_keyname]
                sfq_weights_sq_rand = 1 - cat_neighbors_rand[sfq_keyname]
                counts_gals_rand += np.histogram(radius_neighbors_rand, weights=np.array(sfq_weights_rand), bins=bin_edges)[0]  # smoothed background, assuming bkg is uniform
                counts_gals_ssf_rand += np.histogram(radius_neighbors_rand, weights=np.array(sfq_weights_ssf_rand), bins=bin_edges)[0]
                counts_gals_sq_rand += np.histogram(radius_neighbors_rand, weights=np.array(sfq_weights_sq_rand), bins=bin_edges)[0]
                R_m, R_u = correct_for_masked_area(ra_rand, dec_rand)
                R_m_stack_bkg += R_m
                R_u_stack_bkg += R_u
            else:
                counts_gals_rand += np.zeros(bin_number)

            coord_rand_list.append(coord_rand)
            flag_bkg_rand = 0
    return coord_rand_list, counts_gals_rand, counts_gals_ssf_rand, counts_gals_sq_rand, flag_bkg_rand


def correct_for_masked_area(ra, dec):
    global radius_max
    # correct radial distribution for masked area (if central gal is in partially masked aperture)

    if not correct_masked:
        return np.ones(bin_number), np.ones(bin_number)
    else:
        cat_nomask = cat_random_nomask[abs(cat_random_nomask['RA'] - ra) < radius_max*4 / dis / np.pi * 180]
        cat_nomask = cat_nomask[abs(cat_nomask[dec_key] - dec) < radius_max*4 / dis / np.pi * 180]
        cat_nomask = cat_nomask[SkyCoord(cat_nomask[ra_key] * u.deg, cat_nomask[dec_key] * u.deg).separation(SkyCoord(ra * u.deg, dec * u.deg)).degree < radius_max / dis / np.pi * 180]
        cat_mask = cat_random[abs(cat_random['RA'] - ra) < radius_max*4 / dis / np.pi * 180]
        cat_mask = cat_mask[abs(cat_mask[dec_key] - dec) < radius_max*4 / dis / np.pi * 180]
        cat_mask = cat_mask[SkyCoord(cat_mask[ra_key] * u.deg, cat_mask[dec_key] * u.deg).separation(SkyCoord(ra * u.deg, dec * u.deg)).degree < radius_max / dis / np.pi * 180]

        if len(cat_nomask) == 0:
            return np.zeros(bin_number), np.zeros(bin_number)
        else:
            coord = SkyCoord(ra * u.deg, dec * u.deg)
            coord_nomask = SkyCoord(cat_nomask[ra_key] * u.deg, cat_nomask[dec_key] * u.deg)
            radius_list_nomask = coord_nomask.separation(coord).degree / 180. * np.pi * dis * 1000
            count_nomask = np.histogram(radius_list_nomask, bins=bin_edges)[0]
            count_nomask = np.array(count_nomask).astype(float)
            if len(cat_mask) == 0:
                count_mask = np.zeros(bin_number)
            else:
                coord_mask = SkyCoord(cat_mask[ra_key] * u.deg, cat_mask[dec_key] * u.deg)
                radius_list_mask = coord_mask.separation(coord).degree / 180. * np.pi * dis * 1000
                count_mask = np.histogram(radius_list_mask, bins=bin_edges)[0]
                count_mask = np.array(count_mask).astype(float)
            return count_mask, count_nomask


def spatial_comp(z, masscut_low_comp, masscut_high_comp, ssfq):

    if masscut_low_comp == 9.6:
        masscut_low_comp = 9.5
    elif masscut_low_comp == 10.3 or masscut_low_comp == 10.4:
        masscut_low_comp = 10.2
    if masscut_high_comp == 10.3 or masscut_high_comp == 10.4:
        masscut_high_comp = 10.2

    if not correct_completeness:
        return np.ones(bin_number), np.zeros(bin_number)
    else:
        path_curve = '/home/lejay/completeness_curve_v2/curves_radial_bins/'
        comp_bootstrap = np.genfromtxt(path_curve + 'comp_radial_3.0_all_' + ssfq + '_' + 'sfq_nuvrk' + '_' +str(masscut_low_comp)+'_' + str(masscut_high_comp) + '_' +
                                       str(round(z - 0.1, 1)) + '_' + str(round(z + 0.1, 1)) + '.txt')
        spatial_weight = 1. / comp_bootstrap[:bin_number]
        spatial_weight_err = comp_bootstrap[bin_number:]/comp_bootstrap[:bin_number]**2
        return spatial_weight, spatial_weight_err

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
    return r200/1000  # in Mpc


def cal_error_no_comp_corr(n_sample, n_bkgg, n_bkg_aper, n_centrall, w_comp):
    n_sample[n_sample == 0] = 1
    n_bkgg[n_bkgg == 0] = 1
    sigma_n_sample_pc = cal_error_sat(n_sample, n_centrall, w_comp)
    sigma_b_bkg_pba = cal_error_bkg(n_bkgg, n_bkg_aper, w_comp)
    errors = np.sqrt(sigma_n_sample_pc ** 2 + sigma_b_bkg_pba ** 2)
    return errors


def cal_error_sat(n_sample, n_central, w_comp):
    n_sample[n_sample == 0] = 1
    if len(w_comp) != bin_number:
        w_comp = np.ones(bin_number)
    n_sample_corr = n_sample * w_comp
    errors = (n_sample_corr/n_central)*np.sqrt(1/n_sample_corr)
    return errors


def cal_error_bkg(n_sample, n_bkg_aper, w_comp):
    n_sample[n_sample == 0] = 1
    n_sample_corr = n_sample * np.average(w_comp[-5:])
    if len(w_comp) != bin_number:
        w_comp = np.ones(bin_number)
    sigma_n_sample_corr = np.sqrt(n_sample)*np.average(w_comp[-5:])
    errors = (n_sample_corr/n_bkg_aper)*np.sqrt((sigma_n_sample_corr/n_sample_corr)**2)
    return errors


# ################# START #####################
split_central_mass = False
all_z = False
correct_completeness = False
correct_masked = False
check_edge_flag = True
save_results = True
save_catalogs = True

cen_selection = 'normal'  # pre_select or normal
cat_name = sys.argv[1]  # COSMOS_deep ELAIS_deep XMM-LSS_deep DEEP_deep SXDS_uddd
try:
    sat_z_cut = float(sys.argv[2])
except ValueError:
    sat_z_cut = sys.argv[2]
path = sys.argv[3]
masscut_low = float(sys.argv[4])
masscut_high = float(sys.argv[5])

try:
    masscut_low_host = float(sys.argv[6])
    masscut_high_host = float(sys.argv[7])
    evo_masscut = False
    save_massive_catalogs = True
except ValueError:
    masscut_low_host = 11.0
    masscut_high_host = 13.0
    evo_masscut = True
    save_massive_catalogs = False

csfq = sys.argv[8]  # csf, cq, all
sfq_keyname = sys.argv[9]
rmax = 'fixed'
ssfq_series = ['all', 'ssf', 'sq']
z_bins = [0.6] if all_z else [0.4, 0.6, 0.8, 1.0]
bin_number = 14

if masscut_high <= 1.0:
    correct_completeness = False

# setting binning scheme
if rmax == 'fixed':
    areas = np.array([])
    radius_max = 0.7
    bin_edges = 10 ** np.linspace(np.log10(10), np.log10(radius_max*1000), num=bin_number + 1)
    for i in range(len(bin_edges[:-1])):
        areas = np.append(areas, (bin_edges[i + 1] ** 2 - bin_edges[i] ** 2) * np.pi)

# output run settings and check save path
if path[-1] != '/' and save_results:
    raise NameError('path is not a directory!')
elif save_results:
    print('will save results to ' + path)
    if not os.path.exists(path):
        os.system('mkdir '+path)
else:
    print('will NOT save results!')

cat_type = sys.argv[10]
if cat_type == 'v9':
    cat = Table.read('/home/lejay/catalogs/v9_cats/' + cat_name + '_v9_gal_cut_params_sfq_added.fits')  # galaxy selection already done?
    z_keyname = 'ZPHOT'
    mass_keyname = 'MASS_MED'
    id_keyname = 'ID'
    ra_key = 'RA'
    dec_key = 'DEC'
    save_massive_catalogs = True
    if cat_name == 'XMM-LSS_deep':
        cat = cat[cat['inside_uS'] == True]
    else:
        cat = cat[cat['inside_u'] == True]
    if 'inside_j' in path:
        cat = cat[cat['inside_j'] == True]
    cat = cat[cat['MASK'] == 0]  # unmasked
    cat_gal = cat[cat['OBJ_TYPE'] == 0]  # galaxies

else:
    # COSMOS2020
    cat = Table.read('/home/lejay/catalogs/photoz_cosmos2020_lephare_classic_v1.8_trim_Sfq_phot_Added.fits')  # galaxy selection already done?
    z_keyname = 'zPDF'
    mass_keyname = 'mass_med'
    id_keyname = 'Id'
    ra_key = 'ra'
    dec_key = 'dec'
    save_massive_catalogs = False
    correct_masked = False
    cat = cat[cat['mask'] == 0]  # unmasked
    cat_gal = cat[cat['type'] == 0]  # galaxies

cat_gal = cat_gal[~np.isnan(cat_gal[z_keyname])]
cat_gal = cat_gal[~np.isnan(cat_gal[mass_keyname])]
cat_gal = cat_gal[cat_gal[mass_keyname] > 9.0]
cat_gal = cat_gal[cat_gal['r'] < 25]
if sat_z_cut != 'highz':
    cat_gal = cat_gal[cat_gal[sfq_keyname] >= 0]
    cat_gal = cat_gal[cat_gal[sfq_keyname] <= 1]

# read in random point catalog
if cat_type=='v9':
    cat_random = Table.read('/home/lejay/random_point_cat/'+cat_name + '_random_point.fits')
    cat_random = cat_random[cat_random['inside'] == 0]
    if 'inside_j' in path:
        cat_random = cat_random[cat_random['inside_j'] == 0]
    cat_random_nomask = np.copy(cat_random)
    cat_random = cat_random[cat_random['MASK'] != 0]
else:
    cat_random = Table.read('/home/lejay/random_point_cat/COSMOS2020_random_point_maskadded.fits')
    cat_random_nomask = np.copy(cat_random)
    cat_random = cat_random[cat_random['mask'] == 0]

# ############ main loop ################
cen_cat_dir = 'central_cat/'
print('############ ', cat_type,  masscut_low_host, masscut_high_host)

for z_bin_count, z in enumerate(z_bins):
    z = round(z, 1)
    z_bin_size = 0.3 if all_z else 0.1
    cat_all_z_slice = cat_gal[cat_gal[z_keyname] < 5]
    cat_random_copy = np.copy(cat_random)  # reset random points catalog at each redshift

    # prepare directory for satellite catalogs
    sat_cat_dir = path + cat_name+'_'+str(int(z*10))+'/'
    if not os.path.exists(sat_cat_dir) and save_catalogs:
        os.system('mkdir '+sat_cat_dir)

    print('=============' + str(round(z-z_bin_size, 1)) + '<z<'+str(round(z+z_bin_size, 1))+'================')
    print(csfq, ssfq_series, masscut_low, masscut_high, masscut_low_host, 'evo_masscut =', evo_masscut)
    if cen_selection == 'normal':
        cat_massive_gal = cat_gal[np.logical_and(cat_gal[mass_keyname] > masscut_low_host, cat_gal[mass_keyname] < masscut_high_host)]
        cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal[z_keyname] - z) < z_bin_size]
    elif cen_selection == 'pre_select':
        cat_massive_z_slice = Table.read('/home/lejay/catalogs/central_cat_COSMOS_deep_'+str(round(z,1))+'.fits')
    coord_massive_gal = SkyCoord(np.array(cat_massive_z_slice[ra_key]) * u.deg, np.array(cat_massive_z_slice[dec_key]) * u.deg)
    print('No. of massive gals:', len(cat_massive_z_slice))

    # ### variable initiations ### #
    radial_dist = 0  # radial distribution of satellites
    radial_dist_bkg = 0  # radial distribution of background contamination
    radial_dist_ssf = 0  # radial distribution of satellites
    radial_dist_bkg_ssf = 0  # radial distribution of background contamination
    radial_dist_sq = 0  # radial distribution of satellites
    radial_dist_bkg_sq = 0  # radial distribution of background contamination
    massive_count = 0
    bkg_count = 0
    bin_centers_stack = 0
    companion_count_stack = 0  # used to determine bin_centers
    n_sat = []  # number of satellites per central
    n_bkg = []  # number of bkg contamination per central
    R_m_stack = np.ones(bin_number) * 1e-6  # number of random points (masked) in each bin
    R_u_stack = np.ones(bin_number) * 1e-6  # number of random points (unmasked) in each bin
    R_m_stack_bkg = np.ones(bin_number) * 1e-6  # number of random points (masked, bkg apertures) in each bin
    R_u_stack_bkg = np.ones(bin_number) * 1e-6  # number of random points (unmasked, bkg apertures) in each bin

    # loop for all massive galaxies (potential central galaxy candidate)
    isolated_counts = 0
    isolated_cat = Table(names=cat_massive_z_slice.colnames, dtype=[str(y[0]) for x, y in cat_massive_z_slice.dtype.fields.items()])  # catalog of isolated central galaxies
    nEach = len(cat_massive_z_slice) // nProcs
    if rank == nProcs - 1:
        my_cat_massive_z_slice = cat_massive_z_slice[rank * nEach:]
    else:
        my_cat_massive_z_slice = cat_massive_z_slice[rank * nEach:rank * nEach + nEach]

    print('No. of massive gals in rank'+str(rank)+':', len(my_cat_massive_z_slice))
    for gal in my_cat_massive_z_slice:
        massive_count += 1
        isolation_factor = 10 ** 0
        dis = WMAP9.angular_diameter_distance(gal[z_keyname]).value
        coord_gal = SkyCoord(gal[ra_key] * u.deg, gal[dec_key] * u.deg)
        if check_edge_flag and check_edge(gal[ra_key], gal[dec_key], dis):
            continue

        if rmax == 'r200':
            radius_max = Ms_to_r200(gal['MASS_MED'])  # in Mpc
            bin_edges = 10 ** np.linspace(np.log10(radius_max*1000/70.), np.log10(radius_max*1000), num=bin_number + 1)
            areas = []
            for i in range(len(bin_edges[:-1])):
                areas = np.append(areas, (bin_edges[i + 1] ** 2 - bin_edges[i] ** 2) * np.pi)

        # prepare satellites catalog
        if isinstance(sat_z_cut, float):
            cat_neighbors_z_slice = cat_all_z_slice[abs(cat_all_z_slice[z_keyname] - gal[z_keyname]) < sat_z_cut * 0.044 * (1 + gal[z_keyname])]
        elif sat_z_cut == 'highz':
            cat_neighbors_z_slice = cat_all_z_slice[cat_all_z_slice[z_keyname] > 3.0]

        # spatial cut on satellite sample
        cat_neighbors = cat_neighbors_z_slice[abs(cat_neighbors_z_slice[ra_key] - gal[ra_key]) < radius_max * 4 / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[abs(cat_neighbors[dec_key] - gal[dec_key]) < radius_max * 4 / dis / np.pi * 180] # circular aperture cut
        if len(cat_neighbors) == 0: # skip centrals that have no satellites
            print('No Satellite for', gal[id_keyname])
            continue
        else:
            coord_neighbors = SkyCoord(np.array(cat_neighbors[ra_key]) * u.deg, np.array(cat_neighbors[dec_key]) * u.deg)
            cat_neighbors = cat_neighbors[coord_neighbors.separation(coord_gal).degree < radius_max / dis / np.pi * 180]
            cat_neighbors = cat_neighbors[cat_neighbors[id_keyname] != gal[id_keyname]]  # exclude central galaxy from satellite catalog
            if len(cat_neighbors) == 0:
                print('No Satellite for', gal[id_keyname])
                isolated_cat.add_row(gal)
                isolated_counts += 1
                n_sat.append(0)
                continue

        # isolation cut on central
        if gal[mass_keyname] < np.log10(isolation_factor) + max(cat_neighbors[mass_keyname]):  # no more-massive companions
            continue

        # mass cut on satellite sample
        if masscut_high > 1.0:  # fixed cut (masscut_low)
            cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] > masscut_low]
            cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] < masscut_high]
        else:  # relative cut (masscut_low)
            cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] > gal[mass_keyname] + np.log10(masscut_low)]
            cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] < gal[mass_keyname] + np.log10(masscut_high)]
        if len(cat_neighbors) == 0: # skip centrals that have no satellites
            print('No Satellite for', gal[id_keyname])
            continue

        # evo mass cut
        if evo_masscut and gal[mass_keyname] < 11.4 - 0.16 * gal[z_keyname]:  # if set evolving masscut, apply additional mass cut
            continue

        # cut on central SF/Q
        if csfq == 'csf' and gal[sfq_keyname] < 0.5:  # Q
            continue
        elif csfq == 'cq' and gal[sfq_keyname] >= 0.5:  # SF
            continue

        # skip centrals with no satellite within mass range
        if len(cat_neighbors) == 0:
            isolated_cat.add_row(gal)
            isolated_counts += 1
            n_sat.append(0)
            continue

        # save satellite catalog (no bkg subtraction)
        coord_neighbors = SkyCoord(np.array(cat_neighbors[ra_key]) * u.deg, np.array(cat_neighbors[dec_key]) * u.deg)
        radius_list = coord_neighbors.separation(coord_gal).degree / 180. * np.pi * dis * 1000  # kpc
        if ('all' in ssfq_series) and save_catalogs:
            radius_col = Column(data=radius_list, name='radius', dtype='i4')
            cat_neighbors.add_columns([radius_col])
            cat_neighbors.write(sat_cat_dir + cat_name + '_' + str(gal[id_keyname]) + '_sat.fits', overwrite=True)

        # get mean radius value of satellites in each bin and stack
        # bin_stats = binned_statistic(radius_list, radius_list, statistic='mean', bins=bin_edges)
        bin_stats = np.histogram(radius_list, weights=radius_list, bins=bin_edges)[0] / np.histogram(radius_list, bins=bin_edges)[0]
        bin_centers = bin_stats[0]
        bin_centers = np.nan_to_num(bin_centers)
        for i in range(len(bin_centers)):
            if bin_centers[i] == 0:
                bin_centers[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
        bin_centers_stack += bin_centers * len(radius_list)
        companion_count_stack += len(radius_list)

        # #### CORE FUNCTION: statistics ####
        sfq_weights = np.ones(len(radius_list))
        sfq_weights_ssf = cat_neighbors[sfq_keyname]
        sfq_weights_sq = 1 - cat_neighbors[sfq_keyname]

        # binning all satellite galaxies
        count_binned = np.histogram(radius_list, weights=np.array(sfq_weights), bins=bin_edges)[0]
        count_binned_ssf = np.histogram(radius_list, weights=np.array(sfq_weights_ssf), bins=bin_edges)[0]
        count_binned_sq = np.histogram(radius_list, weights=np.array(sfq_weights_sq), bins=bin_edges)[0]
        sat_counts = np.array(count_binned, dtype='f8')
        sat_counts_ssf = np.array(count_binned_ssf, dtype='f8')
        sat_counts_sq = np.array(count_binned_sq, dtype='f8')

        # mask correction
        R_m, R_u = correct_for_masked_area(gal[ra_key], gal[dec_key])
        R_m_stack += R_m
        R_u_stack += R_u

        # add up all satellite counts for each central
        radial_dist += sat_counts / areas
        radial_dist_ssf += sat_counts_ssf / areas
        radial_dist_sq += sat_counts_sq / areas

        # bkg removal
        coord_random_list, sat_counts_bkg, sat_counts_bkg_ssf, sat_counts_bkg_sq, flag_bkg = bkg(cat_neighbors_z_slice, coord_massive_gal, gal[mass_keyname], dis)
        if flag_bkg == 0:
            radial_dist_bkg += sat_counts_bkg / areas  # bkg completeness correction
            radial_dist_bkg_ssf += sat_counts_bkg_ssf / areas
            radial_dist_bkg_sq += sat_counts_bkg_sq / areas
            bkg_count = bkg_count + 1

        # keep record of cenrtrals and how many satellites a central has
        isolated_counts += 1
        isolated_cat.add_row(gal)
        n_sat.append(sum(sat_counts))

    print('rank', rank, z, sum(n_sat))

    # ######### Collect/Write Results ##################
    # output central catalog
    z_output = 'allz' if all_z else str(z)
    if csfq == 'all' and ('all' in ssfq_series):
        n_sat_col = Column(data=n_sat, name='n_sat', dtype='f8')
        n_bkg_col = Column(data=np.ones(len(n_sat))*sum(radial_dist_bkg*areas)/float(isolated_counts), name='n_bkg', dtype='f8')  # this is avg. bkg count
        if save_massive_catalogs:
            print('No. of isolated centrals in rank '+str(rank)+':', len(isolated_cat))
            isolated_cat.add_columns([n_sat_col, n_bkg_col])
            isolated_cat.write(cen_cat_dir+'isolated_'+cat_name+'_'+str(sat_z_cut)+'_'+str(masscut_low_host)+'_'+z_output+'_'+str(rank)+'_massive.positions.fits', overwrite=True)
    print('Finished gals: ' + str(isolated_counts) + '/' + str(len(my_cat_massive_z_slice)), 'in rank '+str(rank))
    print('Finished bkgs: ' + str(bkg_count), 'in rank '+str(rank))

    # writing results into files
    radial_dist_series = [radial_dist, radial_dist_ssf, radial_dist_sq]  # raw numbers (not corrected for completeness and mask)
    radial_dist_bkg_series = [radial_dist_bkg, radial_dist_bkg_ssf, radial_dist_bkg_sq]  # raw numbers (not corrected for completeness and mask)

    for i, ssfq in enumerate(ssfq_series):
        # normalize number counts by counting area and number of centrals, and correct for mask and completeness
        spatial_weight, spatial_weight_err = spatial_comp(z, masscut_low, masscut_high, ssfq)

        if isolated_counts > 0:
            radial_dist_norm = radial_dist_series[i] * spatial_weight * R_u_stack / R_m_stack / float(isolated_counts)
            radial_dist_bkg_norm = radial_dist_bkg_series[i] * np.max(spatial_weight[-5:]) * R_u_stack_bkg / R_m_stack_bkg / float(bkg_count)
            err = cal_error_no_comp_corr(radial_dist_series[i] * areas * R_u_stack / R_m_stack,
                                     radial_dist_bkg_series[i] * areas * R_u_stack_bkg / R_m_stack_bkg,
                                     isolated_counts, float(bkg_count), spatial_weight) / areas
        else:
            radial_dist_norm = np.zeros(bin_number)
            radial_dist_bkg_norm = np.zeros(bin_number)
            err = np.zeros(bin_number)

        n_central, n_count, n_err = isolated_counts, radial_dist_norm - radial_dist_bkg_norm, err
        result = np.append([n_central], [n_count, n_err])

        # separate sat/bkg number density errors and results
        if isolated_counts>0 and bkg_count>0:
            n_sat_err = cal_error_sat(radial_dist_series[i] * areas, isolated_counts, spatial_weight)/ areas
            n_bkg_err = cal_error_bkg(radial_dist_bkg_series[i] * areas, bkg_count, spatial_weight)/ areas
        else:
            n_sat_err = np.zeros(bin_number)
            n_bkg_err = np.zeros(bin_number)

        result_sat = np.append([n_central], [radial_dist_norm, n_sat_err])
        result_bkg = np.append([n_central], [radial_dist_bkg_norm, n_bkg_err])

        # output result to file
        prefix = '_host_'+str(masscut_low_host) if split_central_mass else ''
        affix = str(sat_z_cut) + '_' + str(masscut_low) + '_' + str(masscut_high) + '_'+ str(csfq) + '_' + str(ssfq) + '_' + z_output + '_' + str(rank) + '.txt'
        filename = path + 'count' + cat_name + prefix + '_' + affix
        filename_sat = path + 'count' + cat_name + prefix + '_sat_' + affix
        filename_bkg = path + 'count' + cat_name + prefix + '_bkg_' + affix
        if save_results:
            np.save(path + 'bin_edges', bin_edges)

            if isolated_counts>0:
                np.save(path + 'bin_centers', bin_centers_stack / companion_count_stack)
            else:
                np.save(path + 'bin_centers', bin_edges[:-1]+np.diff(bin_edges)/2)

            np.savetxt(filename, result, header=time.asctime(time.localtime(time.time())))
            np.savetxt(filename_sat, result_sat, header=time.asctime(time.localtime(time.time())))
            np.savetxt(filename_bkg, result_bkg, header=time.asctime(time.localtime(time.time())))

            if sum(R_u_stack) < 1 and correct_masked:
                print('No massive galaxy with desired satellites!')
            else:
                print(filename + ' saved.')
