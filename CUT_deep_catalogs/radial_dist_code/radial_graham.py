import sys, os
from random import random, sample
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.cosmology import WMAP9
from astropy.table import *
from scipy import stats
from mpi4py import MPI
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
nProcs = comm.Get_size()


def check_edge(ra_rand, dec_rand, dis):
    cat_random_cut = cat_random_copy[abs(cat_random_copy[ra_key] - ra_rand) < 2.5 / dis / np.pi * 180]
    cat_random_cut = cat_random_cut[abs(cat_random_cut[dec_key] - dec_rand) < 2.5 / dis / np.pi * 180]
    coord_random_cut = SkyCoord(np.array(cat_random_cut[ra_key]) * u.deg, np.array(cat_random_cut[dec_key]) * u.deg)
    cat_random_cut = cat_random_cut[coord_random_cut.separation(coord_random_cut).degree < 0.7 / dis / np.pi * 180]

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


def bkg_clustering(cat_neighbors_z_slice_rand, ra_gal, dec_gal, coord_gal, dis):
    global R_m_stack_bkg, R_u_stack_bkg, z

    cat_bkg = cat_neighbors_z_slice_rand[abs(cat_neighbors_z_slice_rand[ra_key] - ra_gal) < 5.0 / dis / np.pi * 180]
    cat_bkg = cat_bkg[abs(cat_bkg[dec_key] - dec_gal) < 5.0 / dis / np.pi * 180]
    coord_bkg = SkyCoord(np.array(cat_bkg[ra_key]) * u.deg, np.array(cat_bkg[dec_key]) * u.deg)
    cat_bkg = cat_bkg[abs(coord_bkg.separation(coord_gal).degree - 0.85 / dis / np.pi * 180) < 0.15 / dis / np.pi * 180]

    if isinstance(masscut_low, float):
        cat_bkg = cat_bkg[cat_bkg[mass_keyname] > masscut_low]
    elif isinstance(masscut_low, str):
        cat_bkg = cat_bkg[cat_bkg[mass_keyname] > gal['MASS_MED'] + np.log10(eval(masscut_low))]

    if isinstance(masscut_high, float):
        cat_bkg = cat_bkg[cat_bkg[mass_keyname] < masscut_high]
    elif isinstance(masscut_high, str):
        cat_bkg = cat_bkg[cat_bkg[mass_keyname] < gal[mass_keyname] + np.log10(eval(masscut_high))]

    R_m, R_u = correct_for_masked_area_clustering(ra_gal, dec_gal)
    R_m_stack_bkg += R_m
    R_u_stack_bkg += R_u

    counts_gals_rand = len(cat_bkg)
    counts_gals_ssf_rand = sum(np.ones(len(cat_bkg))*cat_bkg['sfProb'])
    counts_gals_sq_rand = sum(np.ones(len(cat_bkg))*(1-cat_bkg['sfProb']))
    return counts_gals_rand, counts_gals_ssf_rand, counts_gals_sq_rand


def bkg(cat_neighbors_z_slice_rand, coord_massive_gal_rand, mass_cen, dis):
    global R_m_stack_bkg, R_u_stack_bkg, z
    flag_bkg_rand = -1
    num_before_success = 0
    counts_gals_rand = np.zeros(bin_number)
    counts_gals_ssf_rand = np.zeros(bin_number)
    counts_gals_sq_rand = np.zeros(bin_number)

    coord_rand_list = []
    while flag_bkg_rand == -1:  # get several blank pointing's to estimate background
        id_rand = int(random() * len(cat_random_copy))
        ra_rand = cat_random_copy[id_rand]['RA']
        dec_rand = cat_random_copy[id_rand]['DEC']
        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(ra_rand, dec_rand, unit="deg"), coord_massive_gal_rand, nthneighbor=1)

        num_before_success += 1
        if num_before_success > 100:
            flag_bkg_rand = 1
            break

        if sep2d.degree > 1.4 / dis / np.pi * 180:  # make sure the random pointing is away from any central galaxy (blank)
            if check_edge_flag and check_edge(ra_rand, dec_rand, dis):
                continue

            coord_rand = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)
            cat_neighbors_rand = cat_neighbors_z_slice_rand[abs(cat_neighbors_z_slice_rand[ra_key] - ra_rand) < 2.5 / dis / np.pi * 180]
            cat_neighbors_rand = cat_neighbors_rand[abs(cat_neighbors_rand[dec_key] - dec_rand) < 2.5 / dis / np.pi * 180]
            coord_neighbors_rand = SkyCoord(np.array(cat_neighbors_rand[ra_key]) * u.deg, np.array(cat_neighbors_rand[dec_key]) * u.deg)
            cat_neighbors_rand = cat_neighbors_rand[coord_neighbors_rand.separation(coord_rand).degree < 0.7 / dis / np.pi * 180]

            # make some mass cuts for satellite sample
            if mag_cut:
                cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand[mag_keyname] < magcut_highs[z_bin_count]]
            else:
                if isinstance(masscut_low, float):
                    cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand[mass_keyname] > masscut_low]
                elif isinstance(masscut_low, str):
                    cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand[mass_keyname] > gal['MASS_MED'] + np.log10(eval(masscut_low))]

                if isinstance(masscut_high, float):
                    cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand[mass_keyname] < masscut_high]
                elif isinstance(masscut_high, str):
                    cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand[mass_keyname] < gal[mass_keyname] + np.log10(eval(masscut_high))]

            # exclude bkg apertures that contains galaxies more massive than central (only if not mag_cut)
            if not mag_cut:
                if len(cat_neighbors_rand) > 0:
                    if max(cat_neighbors_rand[mass_keyname]) < mass_cen:
                        num_before_success = 0
                    else:
                        continue
                else:
                    continue

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
                sfq_weights_ssf_rand = cat_neighbors_rand['sfProb']
                sfq_weights_sq_rand = 1 - cat_neighbors_rand['sfProb']

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


def cut_random_cat(cat_rand, coord_list):
    # cut random point catalog to avoid overlapping
    for coord in coord_list:
        coord_rand_list = SkyCoord(np.array(cat_rand[ra_key]) * u.deg, np.array(cat_rand[dec_key]) * u.deg)
        cat_rand = cat_rand[coord_rand_list.separation(coord).degree > 0.7 / dis / np.pi * 180]
    add_to_random_points(coord_list)
    return cat_rand


def add_to_random_points(coord_list):
    # store coord of selected random points just for record
    for coord in coord_list:
        cat_random_points.add_row([coord.ra.value, coord.dec.value, gal[id_keyname]])
    return 0


def correct_for_masked_area_clustering(ra, dec):
    # correct area for normalization if it is partially in masked region
    global areas, area_c
    if not correct_masked:
        return np.ones(bin_number), np.ones(bin_number)
    else:
        cat_nomask = cat_random_nomask[abs(cat_random_nomask[ra_key] - ra) < 5.0 / dis / np.pi * 180]
        cat_nomask = cat_nomask[abs(cat_nomask[dec_key] - dec) < 5.0 / dis / np.pi * 180]
        cat_nomask = cat_nomask[abs(SkyCoord(np.array(cat_nomask[ra_key]) * u.deg, np.array(cat_nomask[dec_key]) * u.deg).separation(coord_gal).degree - 0.85 / dis / np.pi * 180) < 0.15 / dis / np.pi * 180]

        cat_mask = cat_random[abs(cat_random[ra_key] - ra) < 5.0 / dis / np.pi * 180]
        cat_mask = cat_mask[abs(cat_mask[dec_key] - dec) < 5.0 / dis / np.pi * 180]
        cat_mask = cat_mask[abs(SkyCoord(np.array(cat_mask[ra_key]) * u.deg, np.array(cat_mask[dec_key]) * u.deg).separation(coord_gal).degree - 0.85 / dis / np.pi * 180) < 0.15 / dis / np.pi * 180]

        if len(cat_nomask) == 0:
            return np.zeros(bin_number), np.zeros(bin_number)
        else:
            count_nomask = len(cat_nomask)
            if len(cat_mask) == 0:
                count_mask = np.zeros(bin_number)
            else:
                count_mask = len(cat_mask)

            return count_mask*areas/area_c, count_nomask*areas/area_c


def correct_for_masked_area(ra, dec):
    # correct radial distribution for masked area (if central gal is in partially masked aperture)

    if not correct_masked:
        return np.ones(bin_number), np.ones(bin_number)
    else:
        cat_nomask = cat_random_nomask[abs(cat_random_nomask['RA'] - ra) < 2.5 / dis / np.pi * 180]
        cat_nomask = cat_nomask[abs(cat_nomask[dec_key] - dec) < 2.5 / dis / np.pi * 180]
        cat_nomask = cat_nomask[SkyCoord(cat_nomask[ra_key] * u.deg, cat_nomask[dec_key] * u.deg).separation
                            (SkyCoord(ra * u.deg, dec * u.deg)).degree < 0.7 / dis / np.pi * 180]

        cat_mask = cat_random[abs(cat_random['RA'] - ra) < 2.5 / dis / np.pi * 180]
        cat_mask = cat_mask[abs(cat_mask[dec_key] - dec) < 2.5 / dis / np.pi * 180]
        cat_mask = cat_mask[SkyCoord(cat_mask[ra_key] * u.deg, cat_mask[dec_key] * u.deg).separation
                        (SkyCoord(ra * u.deg, dec * u.deg)).degree < 0.7 / dis / np.pi * 180]

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
        path_curve = '~/curves_graham/'
        try:
            comp_bootstrap = np.genfromtxt(
                path_curve + 'comp_bootstrap_all_' + ssfq + '_'+str(masscut_low_comp)+'_' + str(masscut_high_comp) + '_'
                + str(round(z - 0.1, 1)) + '_' + str(round(z + 0.1, 1)) + '.txt')
        except IOError:
            comp_bootstrap = np.genfromtxt(
                path_curve + 'comp_bootstrap_all_' + ssfq + '_' + str(round(z - 0.1, 1)) + '_'
                + str(round(z + 0.1, 1)) + '_'+str(masscut_low_comp)+'_' + str(masscut_high_comp) + '.txt')

        spatial_weight = 1. / np.nanmedian(comp_bootstrap, axis=0)
        spatial_weight_err = np.nanstd(comp_bootstrap, axis=0)/np.nanmedian(comp_bootstrap, axis=0)**2

        return spatial_weight, spatial_weight_err


def cal_error(n_sample, n_bkgg, n_centrall, n_bkg_aper, w_comp, w_comp_err):
    '''
        n_sample:
            number of potential satellites (stacked)
        n_bkgg:
            number of bkg objects (stacked)
        w_comp:
            completeness correction factor
        w_comp_err:
            error estimated for w_comp

        n_sample and n_bkgg should be satellite/bkg counts raw number.
    '''

    n_sample[n_sample == 0] = 1
    n_bkgg[n_bkgg == 0] = 1

    if len(w_comp) != bin_number:
        w_comp = np.ones(bin_number)
        w_comp_err = np.zeros(bin_number)

    n_sample_corr = n_sample / w_comp  # correct n_sample for completeness
    sigma_n_sample_corr = (n_sample / w_comp) * np.sqrt(1 / n_sample + (w_comp_err / w_comp) ** 2)

    n_bkg_corr = n_bkgg / np.average(w_comp[-5:])  # correct n_bkgg for completeness

    sigma_n_sample_pc = n_sample_corr/n_centrall * np.sqrt((sigma_n_sample_corr/n_sample_corr)**2 + 1/n_centrall)  # per central
    sigma_b_bkg_pba = n_bkg_corr/n_bkg_aper * np.sqrt(1/n_bkgg + 1/n_bkg_aper)  # per bkg aperture

    errors = np.sqrt(sigma_n_sample_pc**2 + sigma_b_bkg_pba**2)
    return errors


def cal_error_no_comp_corr(n_sample, n_bkgg, n_bkg_aper, n_centrall):
    n_sample[n_sample == 0] = 1
    n_bkgg[n_bkgg == 0] = 1

    sigma_n_sample_pc = n_sample / n_centrall * np.sqrt(1/ n_sample + 1 / n_centrall)  # per central
    sigma_b_bkg_pba = n_bkgg / n_bkg_aper * np.sqrt(1 / n_bkgg + 1 / n_bkg_aper)  # per bkg aperture

    errors = np.sqrt(sigma_n_sample_pc ** 2 + sigma_b_bkg_pba ** 2)
    return errors


def cal_error_old(n_sample, n_bkg, n_central, w_comp, w_comp_err):

    n_comp = np.array(n_sample - n_bkg).astype(float)  # number of satellites in each bin
    n_comp[n_comp==0] = 1  # when calculating error, if no satellite in one bin, use n=1 in that bin to avoid ValueError

    if len(w_comp) != bin_number:
        w_comp = np.ones(bin_number)
        w_comp_err = np.zeros(bin_number)

    sigma_n_sample_corr = (n_comp/w_comp) * np.sqrt(1/n_comp+(w_comp_err/w_comp)**2)
    sigma_n_bkg = np.sqrt(n_bkg)
    sigma_n_comp = np.sqrt(sigma_n_sample_corr**2+sigma_n_bkg**2)
    sigma_n_central = np.array(np.sqrt(n_central)).astype(float)

    errors = (n_comp/n_central)*np.sqrt((sigma_n_comp/n_comp)**2+(sigma_n_central/n_central)**2)
    return errors


def cal_error_sat(n_sample, n_central, w_comp, w_comp_err):
    n_sample[n_sample == 0] = 1
    n_sample_corr = n_sample / w_comp

    if len(w_comp) != bin_number:
        w_comp = np.ones(bin_number)
        w_comp_err = np.zeros(bin_number)

    sigma_n_sample_corr = (n_sample/w_comp) * np.sqrt(1/n_sample+(w_comp_err/w_comp)**2)

    errors = (n_sample_corr/n_central)*np.sqrt((sigma_n_sample_corr/n_sample_corr)**2 + 1/n_central)
    return errors


def cal_error_bkg(n_sample, n_bkg_aper, w_comp):
    n_sample[n_sample == 0] = 1
    n_sample_corr = n_sample / np.average(w_comp[-5:])

    if len(w_comp) != bin_number:
        w_comp = np.ones(bin_number)

    sigma_n_sample_corr = np.sqrt(n_sample)/np.average(w_comp[-5:])
    errors = (n_sample_corr/n_bkg_aper)*np.sqrt((sigma_n_sample_corr/n_sample_corr)**2 + 1/n_bkg_aper)
    return errors


# ################# START #####################
cat_name = sys.argv[1]  # COSMOS_deep COSMOS_uddd ELAIS_deep XMM-LSS_deep DEEP_deep SXDS_uddd
mode = sys.argv[2]  # 'count' or 'mass'

# run settings
split_central_mass = False
all_z = False
correct_completeness = True
correct_masked = True
check_edge_flag = True
save_results = True
save_catalogs = False
save_massive_catalogs = True
evo_masscut = False
no_z_cut = False  # ignore redshift selection criteria for satellite selection
mag_cut = False  # if false then it's mass_cut == True
remove_contam = 0  # 0-all (no remove), 1-goodz, 2-badz 3-remove all sats that do not satisfy both selection  |z_A - z_T|>0.05*z_Ts
bkg_option = 'random'  # 'random' or 'clustering'
sat_z_cut = 4.5
z_bins = [0.6] if all_z else [0.4, 0.6, 0.8]
csfq = 'all'  # csf, cq, all
ssfq_series = ['all']
masscut_low = 9.5
masscut_high = 13.0
masscut_low_host = 11.0 if evo_masscut else 11.15
masscut_high_host = 13.0
sfprob_cut_low = 0.5
sfprob_cut_high = 0.5

# magnitude cuts
mag_keynames = ['ri', 'i', 'z']
magcut_highs = [24, 24.91, 25.55]

# setting binning scheme
bin_number = 14
bin_edges = 10 ** np.linspace(1.0, np.log10(700), num=bin_number + 1)
area_c = (1000**2-700**2) * np.pi  # in kpc
areas = np.array([])
for i in range(len(bin_edges[:-1])):
    areas = np.append(areas, (bin_edges[i + 1] ** 2 - bin_edges[i] ** 2) * np.pi)

# read in data catalog
path = 'total_sample_v6cat/'
print('start reading catalogs ...')
cat = Table.read('~/catalogs/v8_cats/'+cat_name+'_v8_gal_cut_params.fits')
cat.add_column(Column(data=np.ones(len(cat)), name='sfProb', dtype='f8'))
cat = cat[cat['MASK']==0]
if cat_name == 'XMM-LSS_deep':
    cat = cat[cat['inside_uS'] == True]
else:
    cat = cat[cat['inside_u'] == True]
z_keyname = 'ZPHOT'
mass_keyname = 'MASS_MED'
id_keyname = 'ID'
ra_key = 'RA'
dec_key = 'DEC'
cat = cat[~np.isnan(cat[z_keyname])]
cat_gal = cat[cat['OBJ_TYPE'] == 0]  # galaxies

# initial cuts
if mag_cut:
    for mag_key in mag_keynames:
        cat_gal = cat_gal[cat_gal[mag_key] > 0]
else:
    cat_gal = cat_gal[cat_gal[mass_keyname] > 9.0]

# output run settings and check save path
if path[-1] != '/' and save_results:
    raise NameError('path is not a directory!')
elif save_results:
    if os.path.exists(path):
        print('will save results to ' + path)
    else:
        raise FileNotFoundError('No such path!')
else:
    print('will NOT save results!')

# read-in random point catalog
cat_random = Table.read('~/random_point_cat'+cat_name + '_random_point.fits')
try:
    cat_random = cat_random[cat_random['inside_u'] == True]
except KeyError:
    cat_random = cat_random[cat_random['inside_uS'] == True]
cat_random_nomask = np.copy(cat_random)
cat_random = cat_random[cat_random['MASK'] == 0]
cat_random_points = Table(names=('RA', 'DEC', 'GAL_ID'))   # to store position of selected random apertures

# ############ main loop ################
for z_bin_count, z in enumerate(z_bins):
    z = round(z, 1)
    z_bin_size = 0.3 if all_z else 0.1
    mag_keyname = mag_keynames[z_bin_count]

    # prepare directory for satellite catalogs
    sat_cat_dir = path + cat_name+'_'+str(z*10)+'/'
    if not os.path.exists(sat_cat_dir):
        os.system('mkdir '+sat_cat_dir)

    print('=============' + str(round(z-z_bin_size, 1)) + '<z<'+str(round(z+z_bin_size, 1))+'================')
    print(mode, csfq, ssfq_series, masscut_low, masscut_high, masscut_low_host, 'evo_masscut =', evo_masscut)
    cat_random_copy = np.copy(cat_random)  # reset random points catalog at each redshift

    # select centrals that satisfy both old and new selection
    cat_massive_gal = cat_gal[np.logical_and(cat_gal[mass_keyname] > masscut_low_host, cat_gal[mass_keyname] < masscut_high_host)]
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal[z_keyname] - z) < z_bin_size]
    print('No. of massive gals:', len(cat_massive_z_slice))
    coord_massive_gal = SkyCoord(np.array(cat_massive_z_slice[ra_key]) * u.deg, np.array(cat_massive_z_slice[dec_key]) * u.deg)

    # initial redshift slice to reduce computation
    if no_z_cut:
        cat_all_z_slice = cat_gal
    else:
        cat_all_z_slice = cat_gal[abs(cat_gal[z_keyname] - z) < 0.7]

    # ### variable initiations ### #
    radial_dist = 0  # radial distribution of satellites
    radial_dist_bkg = 0  # radial distribution of background contamination
    radial_dist_ssf = 0  # radial distribution of satellites
    radial_dist_bkg_ssf = 0  # radial distribution of background contamination
    radial_dist_sq = 0  # radial distribution of satellites
    radial_dist_bkg_sq = 0  # radial distribution of background contamination
    radial_count_dist = 0  # radial number density distribution of satellites (used in mass mode)
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
    isolated_counts2 = 0
    isolated_cat = Table(names=cat_massive_z_slice.colnames, dtype=[str(y[0]) for x, y in cat_massive_z_slice.dtype.fields.items()])  # catalog of isolated central galaxies

    nEach = len(cat_massive_z_slice) // nProcs
    if rank == nProcs - 1:
        my_cat_massive_z_slice = cat_massive_z_slice[rank * nEach:]
    else:
        my_cat_massive_z_slice = cat_massive_z_slice[rank * nEach:rank * nEach + nEach]

    for gal in cat_massive_z_slice:
        massive_count += 1
        isolation_factor = 10 ** 0  #(1*(gal['MASS_MED']-gal['MASS_INF']))

        print('Progress:' + str(massive_count) + '/' + str(len(cat_massive_z_slice)), end='\r')
        dis = WMAP9.angular_diameter_distance(gal[z_keyname]).value
        coord_gal = SkyCoord(gal[ra_key] * u.deg, gal[dec_key] * u.deg)
        if check_edge_flag and check_edge(gal[ra_key], gal[dec_key], dis):
            continue

        # prepare neighbors catalog
        if no_z_cut:
            cat_neighbors_z_slice = cat_gal  # NO redshift cut in this case
        else:
            cat_neighbors_z_slice = cat_all_z_slice[abs(cat_all_z_slice[z_keyname] - gal[z_keyname]) < sat_z_cut * 0.044 * (1 + gal[z_keyname])]

        # initial rectangular spatial cut
        cat_neighbors = cat_neighbors_z_slice[abs(cat_neighbors_z_slice[ra_key] - gal[ra_key]) < 2.5 / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[abs(cat_neighbors[dec_key] - gal[dec_key]) < 2.5 / dis / np.pi * 180]

        # circular aperture cut, skip centrals that have no satellites
        if len(cat_neighbors) == 0:
            print('No Satellite for', gal[id_keyname])
            continue
        else:
            coord_neighbors = SkyCoord(np.array(cat_neighbors[ra_key]) * u.deg, np.array(cat_neighbors[dec_key]) * u.deg)
            cat_neighbors = cat_neighbors[coord_neighbors.separation(coord_gal).degree < 0.7 / dis / np.pi * 180]

            cat_neighbors = cat_neighbors[cat_neighbors[id_keyname] != gal[id_keyname]]  # exclude central galaxy from satellite catalog
            if len(cat_neighbors) == 0:
                print('No Satellite for', gal[id_keyname])
                isolated_cat.add_row(gal)
                n_sat.append(0)
                continue

        # isolation cut on central
        if gal[mass_keyname] < np.log10(isolation_factor) + max(cat_neighbors[mass_keyname]):  # no more-massive companions
            continue

        # evo mass cut
        if evo_masscut and gal[mass_keyname] < 11.23 - 0.16 * gal[z_keyname]:  # if set evolving masscut, apply additional mass cut
            continue

        # cut on central SF/Q
        if csfq == 'csf' and gal['sfProb'] < sfprob_cut_high:
            continue
        elif csfq == 'cq' and gal['sfProb'] >= sfprob_cut_low:
            continue

        # cut on satellite sample (mass cut, ssfr cut)
        if mag_cut:
            cat_neighbors = cat_neighbors[cat_neighbors[mag_keyname] < magcut_highs[z_bin_count]]
        else:
            # masscut_low
            if isinstance(masscut_low, float):  # fixed cut
                cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] > masscut_low]
            elif isinstance(masscut_low, str):  # relative cut
                cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] > gal[mass_keyname]+np.log10(eval(masscut_low))]

            # masscut_high
            if isinstance(masscut_high, float):  # fixed cut
                cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] < masscut_high]
            elif isinstance(masscut_high, str):  # relative cut
                cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] < gal[mass_keyname]+np.log10(eval(masscut_high))]

        # skip centrals with no satellite within mass range
        if len(cat_neighbors) == 0:
            isolated_cat.add_row(gal)
            isolated_counts2 += 1
            n_sat.append(0)
            continue

        # save satellite catalog (no bkg subtraction)
        coord_neighbors = SkyCoord(np.array(cat_neighbors[ra_key]) * u.deg, np.array(cat_neighbors[dec_key]) * u.deg)
        radius_list = coord_neighbors.separation(coord_gal).degree / 180. * np.pi * dis * 1000  # kpc
        if ('all' in ssfq_series) and save_catalogs:
            radius_col = Column(data=radius_list, name='radius', dtype='i4')
            cat_neighbors.add_columns([radius_col])
            cat_neighbors.write(sat_cat_dir + cat_name + '_' + str(gal[id_keyname]) + '_sat.fits', overwrite=True)

        # get mean radius value of companions in each bin and stack
        bin_stats = stats.binned_statistic(radius_list, radius_list, statistic='mean', bins=bin_edges)
        bin_centers = bin_stats[0]
        bin_centers = np.nan_to_num(bin_centers)
        for i in range(len(bin_centers)):
            if bin_centers[i] == 0:
                bin_centers[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
        bin_centers_stack += bin_centers * len(radius_list)
        companion_count_stack += len(radius_list)

        # #### CORE FUNCTION: statistics ####
        sfq_weights = np.ones(len(radius_list))
        sfq_weights_ssf = cat_neighbors['sfProb']
        sfq_weights_sq = 1 - cat_neighbors['sfProb']

        # binning all satellite galaxies
        count_binned = np.histogram(radius_list, weights=np.array(sfq_weights), bins=bin_edges)[0]
        count_binned_ssf = np.histogram(radius_list, weights=np.array(sfq_weights_ssf), bins=bin_edges)[0]
        count_binned_sq = np.histogram(radius_list, weights=np.array(sfq_weights_sq), bins=bin_edges)[0]

        # count of satellite galaxies in each bin
        sat_counts = np.array(count_binned, dtype='f8')
        sat_counts_ssf = np.array(count_binned_ssf, dtype='f8')
        sat_counts_sq = np.array(count_binned_sq, dtype='f8')

        # mask correction
        R_m, R_u = correct_for_masked_area(gal[ra_key], gal[dec_key])
        R_m_stack += R_m
        R_u_stack += R_u

        # add up all satellite counts for each central
        radial_dist += sat_counts
        radial_dist_ssf += sat_counts_ssf
        radial_dist_sq += sat_counts_sq

        # bkg removal
        if bkg_option == 'random':
            if not mag_cut:
                coord_random_list, sat_counts_bkg, sat_counts_bkg_ssf, sat_counts_bkg_sq, flag_bkg = bkg(
                    cat_neighbors_z_slice, coord_massive_gal, gal[mass_keyname], dis)
            else:
                coord_random_list, sat_counts_bkg, sat_counts_bkg_ssf, sat_counts_bkg_sq, flag_bkg = bkg(
                    cat_neighbors_z_slice, coord_massive_gal, 11, dis)

            if flag_bkg == 0:
                radial_dist_bkg += sat_counts_bkg   # bkg completeness correction
                radial_dist_bkg_ssf += sat_counts_bkg_ssf
                radial_dist_bkg_sq += sat_counts_bkg_sq
                bkg_count = bkg_count + 1
        else:
            sat_counts_bkg, sat_counts_bkg_ssf, sat_counts_bkg_sq = bkg_clustering(cat_neighbors_z_slice, gal['RA'], gal['DEC'], coord_gal, dis)
            radial_dist_bkg += sat_counts_bkg * np.ones(bin_number) * areas/area_c   # bkg completeness correction
            radial_dist_bkg_ssf += sat_counts_bkg_ssf * areas/area_c
            radial_dist_bkg_sq += sat_counts_bkg_sq * areas/area_c
            bkg_count = bkg_count + 1

        # keep record of cenrtrals and how many satellites a central has
        isolated_counts2 += 1
        isolated_cat.add_row(gal)
        n_sat.append(sum(sat_counts))

    # ######### Aggregation of Results ##################
    # output central catalog
    z_output = 'allz' if all_z else str(z)
    if csfq == 'all' and ('all' in ssfq_series):
        n_sat_col = Column(data=n_sat, name='n_sat', dtype='i4')
        n_bkg_col = Column(data=np.ones(len(n_sat))*sum(radial_dist_bkg)/float(isolated_counts2), name='n_bkg', dtype='i4')  # this is avg. bkg count
        if save_massive_catalogs:
            isolated_cat.write(sat_cat_dir+'isolated_'+cat_name+'_'+str(masscut_low_host)+'_'+z_output+'_massive.positions.fits', overwrite=True)

    print('Finished gals: ' + str(isolated_counts2) + '/' + str(len(cat_massive_z_slice)))
    print('Finished bkgs: ' + str(bkg_count))
    cat_random_points.write('random_points_' + cat_name + '_' + str(z) + '.fits', overwrite=True)

    # writing results into files
    radial_dist_series = [radial_dist, radial_dist_ssf, radial_dist_sq]  # raw numbers (not corrected for completeness and mask)
    radial_dist_bkg_series = [radial_dist_bkg, radial_dist_bkg_ssf, radial_dist_bkg_sq]  # raw numbers (not corrected for completeness and mask)
    for i, ssfq in enumerate(ssfq_series):
        # normalize number counts by counting area and number of centrals, and correct for mask and completeness
        spatial_weight, spatial_weight_err = spatial_comp(z, masscut_low, masscut_high, ssfq)
        radial_dist_norm = radial_dist_series[i] * spatial_weight * R_u_stack / R_m_stack / float(isolated_counts2) / areas
        radial_dist_bkg_norm = radial_dist_bkg_series[i] * np.average(spatial_weight[-5:]) * R_u_stack_bkg / R_m_stack_bkg / float(bkg_count) / areas

        err = cal_error_no_comp_corr(radial_dist_series[i] * R_u_stack / R_m_stack,
                        radial_dist_bkg_series[i] * R_u_stack_bkg / R_m_stack_bkg, isolated_counts2, float(bkg_count)) / areas

        n_central, n_count, n_err = isolated_counts2, radial_dist_norm - radial_dist_bkg_norm, err
        result = np.append([n_central], [n_count, n_err])

        # sat/bkg number density errors and results
        n_sat_err = cal_error_sat(radial_dist_series[i], isolated_counts2, spatial_weight, spatial_weight_err) / areas
        n_bkg_err = cal_error_bkg(radial_dist_bkg_series[i], bkg_count, spatial_weight) / areas
        result_sat = np.append([n_central], [radial_dist_norm, n_sat_err])
        result_bkg = np.append([n_central], [radial_dist_bkg_norm, n_bkg_err])

        # output result to file
        prefix = '_host_'+str(masscut_low_host) if split_central_mass else ''
        affix = str(masscut_low) + '_' + str(csfq) + '_' + str(ssfq) + '_' + z_output + '_' + str(rank) + '.txt'

        filename = path + str(mode) + cat_name + prefix + '_' + affix
        filename_sat = path + str(mode) + cat_name + prefix + '_sat_' + affix
        filename_bkg = path + str(mode) + cat_name + prefix + '_bkg_' + affix
        if save_results:
            np.save(path + 'bin_edges', bin_edges)
            np.save(path + 'bin_centers', bin_centers_stack / companion_count_stack)
            if sum(R_u_stack) < 1 and correct_masked:
                np.savetxt(filename, result)
                print('No massive galaxy with desired satellites!')
            else:
                print(filename)
                np.savetxt(filename, result)
                np.savetxt(filename_sat, result_sat)
                np.savetxt(filename_bkg, result_bkg)
