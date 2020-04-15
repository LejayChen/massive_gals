import sys, os
from random import random
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.cosmology import WMAP9
from astropy.table import *
from scipy import stats


def check_edge(ra_rand, dec_rand, dis):
    cat_random_cut = cat_random_copy[abs(cat_random_copy['RA'] - ra_rand) < 2.5 / dis / np.pi * 180]
    cat_random_cut = cat_random_cut[abs(cat_random_cut['DEC'] - dec_rand) < 2.5 / dis / np.pi * 180]
    coord_random_cut = SkyCoord(cat_random_cut['RA'] * u.deg, cat_random_cut['DEC'] * u.deg)
    cat_random_cut = cat_random_cut[coord_random_cut.separation(coord_gal).degree < 0.7 / dis / np.pi * 180]

    try:
        ra_ratio = len(cat_random_cut[cat_random_cut['RA'] < ra_rand])/len(cat_random_cut[cat_random_cut['RA'] > ra_rand])
        dec_ratio = len(cat_random_cut[cat_random_cut['DEC'] < dec_rand])/len(cat_random_cut[cat_random_cut['DEC'] > dec_rand])
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

    cat_bkg = cat_neighbors_z_slice_rand[abs(cat_neighbors_z_slice_rand['RA'] - ra_gal) < 5.0 / dis / np.pi * 180]
    cat_bkg = cat_bkg[abs(cat_bkg['DEC'] - dec_gal) < 5.0 / dis / np.pi * 180]
    coord_bkg = SkyCoord(cat_bkg['RA'] * u.deg, cat_bkg['DEC'] * u.deg)
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
            flag_bkg = 1
            break

        if sep2d.degree > 1.4 / dis / np.pi * 180:  # make sure the random pointing is away from any central galaxy (blank)

            if check_edge(ra_rand, dec_rand, dis):
                continue

            coord_rand = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)
            cat_neighbors_rand = cat_neighbors_z_slice_rand[abs(cat_neighbors_z_slice_rand['RA'] - ra_rand) < 2.5 / dis / np.pi * 180]
            cat_neighbors_rand = cat_neighbors_rand[abs(cat_neighbors_rand['DEC'] - dec_rand) < 2.5 / dis / np.pi * 180]
            coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)
            cat_neighbors_rand = cat_neighbors_rand[coord_neighbors_rand.separation(coord_rand).degree < 0.7 / dis / np.pi * 180]

            # make some mass cuts for satellite sample
            if isinstance(masscut_low, float):
                cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand[mass_keyname] > masscut_low]
            elif isinstance(masscut_low, str):
                cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand[mass_keyname] > gal['MASS_MED'] + np.log10(eval(masscut_low))]

            if isinstance(masscut_high, float):
                cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand[mass_keyname] < masscut_high]
            elif isinstance(masscut_high, str):
                cat_neighbors_rand = cat_neighbors_rand[
                    cat_neighbors_rand[mass_keyname] < gal[mass_keyname] + np.log10(eval(masscut_high))]

            if ('all' in ssfq_series) and save_catalogs:
                cat_neighbors_rand.write(sat_cat_dir + cat_name + '_' + str(gal[id_keyname]) + '_bkg.fits', overwrite=True)

            # exclude bkg apertures that contains galaxies more massive than central
            if len(cat_neighbors_rand) > 0:
                if max(cat_neighbors_rand[mass_keyname]) < mass_cen:
                    num_before_success = 0
                else:
                    continue
            else:
                continue

            coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)
            radius_neighbors_rand = coord_neighbors_rand.separation(coord_rand).degree / 180. * np.pi * dis * 1000
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

    return coord_rand_list, counts_gals_rand, counts_gals_ssf_rand, counts_gals_sq_rand, flag_bkg


def cut_random_cat(cat_rand, coord_list):
    # cut random point catalog to avoid overlapping
    for coord in coord_list:
        coord_rand_list = SkyCoord(cat_rand['RA'] * u.deg, cat_rand['DEC'] * u.deg)
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
        cat_nomask = cat_random_nomask[abs(cat_random_nomask['RA'] - ra) < 5.0 / dis / np.pi * 180]
        cat_nomask = cat_nomask[abs(cat_nomask['DEC'] - dec) < 5.0 / dis / np.pi * 180]
        cat_nomask = cat_nomask[abs(SkyCoord(cat_nomask['RA'] * u.deg, cat_nomask['DEC'] * u.deg).separation(coord_gal).degree - 0.85 / dis / np.pi * 180) < 0.15 / dis / np.pi * 180]

        cat_mask = cat_random[abs(cat_random['RA'] - ra) < 5.0 / dis / np.pi * 180]
        cat_mask = cat_mask[abs(cat_mask['DEC'] - dec) < 5.0 / dis / np.pi * 180]
        cat_mask = cat_mask[abs(SkyCoord(cat_mask['RA'] * u.deg, cat_mask['DEC'] * u.deg).separation(coord_gal).degree - 0.85 / dis / np.pi * 180) < 0.15 / dis / np.pi * 180]

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
    # correct area for normalization if it is partially in masked region
    if not correct_masked:
        return np.ones(bin_number), np.ones(bin_number)
    else:
        cat_nomask = cat_random_nomask[abs(cat_random_nomask['RA'] - ra) < 2.5 / dis / np.pi * 180]
        cat_nomask = cat_nomask[abs(cat_nomask['DEC'] - dec) < 2.5 / dis / np.pi * 180]
        cat_nomask = cat_nomask[SkyCoord(cat_nomask['RA'] * u.deg, cat_nomask['DEC'] * u.deg).separation
                            (SkyCoord(ra * u.deg, dec * u.deg)).degree < 0.7 / dis / np.pi * 180]

        cat_mask = cat_random[abs(cat_random['RA'] - ra) < 2.5 / dis / np.pi * 180]
        cat_mask = cat_mask[abs(cat_mask['DEC'] - dec) < 2.5 / dis / np.pi * 180]
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


def spatial_comp(z, masscut_low_comp, masscut_high_comp, ssfq):
    if masscut_low_comp == 9.6:
        masscut_low_comp = 9.5
    elif masscut_low_comp == 10.3:
        masscut_low_comp = 10.2
    if masscut_high_comp == 10.3:
        masscut_high_comp = 10.2

    if not correct_completeness:
        return np.ones(bin_number), np.zeros(bin_number)
    else:
        path_curve = '/Users/lejay/research/massive_gals/completeness_curve/curves_graham/'
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


def cal_error2(n_sample, n_central, w_comp, w_comp_err):
    n_comp = np.array(n_sample).astype(float)
    n_comp[n_comp == 0] = 1

    if len(w_comp) != bin_number:
        w_comp = np.ones(bin_number)
        w_comp_err = np.zeros(bin_number)

    sigma_n_sample_ori = np.sqrt(n_sample/w_comp)
    sigma_n_sample_corr = np.sqrt(sigma_n_sample_ori**2*w_comp**2 + (n_sample/w_comp)**2*w_comp_err**2)
    sigma_n_comp = sigma_n_sample_corr
    sigma_n_central = np.array(np.sqrt(n_central)).astype(float)

    errors = (n_comp/n_central)*np.sqrt((sigma_n_comp/n_comp)**2+(sigma_n_central/n_central)**2)
    return errors


# ################# START #####################
cat_name = sys.argv[1]  # COSMOS_deep COSMOS_uddd ELAIS_deep XMM-LSS_deep DEEP_deep SXDS_uddd
mode = sys.argv[2]  # 'count' or 'mass'

# run settings
split_central_mass = False
all_z = False
correct_completeness = True
correct_masked = False
save_results = True
save_catalogs = True
save_massive_catalogs = True
evo_masscut = False
bkg_option = 'random'  # 'random' or 'clustering'
sat_z_cut = 4.5
z_bins = [0.6] if all_z else [0.4, 0.6, 0.8]
csfq = 'all'  # csf, cq, all
ssfq_series = ['all']
masscut_low = 9.5
masscut_high = 13.0
masscut_low_host = 11.0 if evo_masscut else 11.15
masscut_high_host = 13.0
isolation_factor = 1
sfprob_cut_low = 0.5
sfprob_cut_high = 0.5

# setting binning scheme
bin_number = 14
bin_edges = 10 ** np.linspace(1.0, np.log10(700), num=bin_number + 1)
area_c = (1000**2-700**2) * np.pi  # in kpc
areas = np.array([])
for i in range(len(bin_edges[:-1])):
    areas = np.append(areas, (bin_edges[i + 1] ** 2 - bin_edges[i] ** 2) * np.pi)

# read in data catalog
cat_type = 'old'
params = 'old'
massive_selection = 'normal'  # old, new, both-central,  or normal
# path = 'total_sample_matched_cat_massive_'+massive_selection+'_'+params+'_params_0405/'
path = 'total_sample_nomask_corr_0412/'
print('start reading catalogs ...')
if cat_type == 'old':
    if cat_name == 'SXDS_uddd':
        cat_name = 'SXDS3_uddd'
    cat = Table.read('s16a_' + cat_name + '_masterCat.fits')
    z_keyname = 'zKDEPeak'
    mass_keyname = 'MASS_MED'
    id_keyname = 'NUMBER'
    cat = cat[~np.isnan(cat[z_keyname])]
    cat = cat[cat[z_keyname] < 1.3]
    cat_gal = cat[np.logical_and(cat['preds_median'] < 0.89, cat['inside'] == True)]
elif cat_type == 'matched':
    cat = Table.read('s16a_' + cat_name + '_masterCat_newz3.fits')
    cat = cat[cat['matched']==True]
    id_keyname = 'NUMBER'
    if params == 'new':
        z_keyname = 'Z_BEST_BC03'
        mass_keyname = 'MASS_MED_new'
        cat = cat[~np.isnan(cat[z_keyname])]
        cat = cat[cat[z_keyname] < 1.3]
        cat_gal = cat[cat['CLASS'] < 10]
    else:
        z_keyname = 'zKDEPeak'
        mass_keyname = 'MASS_MED'
        cat = cat[~np.isnan(cat[z_keyname])]
        cat = cat[cat[z_keyname] < 1.3]
        cat_gal = cat[np.logical_and(cat['preds_median'] < 0.89, cat['inside'] == True)]
elif cat_type == 'new':
    cat = Table.read('UV_CUT_CLAUDS_HSC_S16A_' + cat_name + '.fits')
    z_keyname = 'Z_BEST_BC03'
    mass_keyname = 'MASS_MED'
    id_keyname = 'ID'
    cat = cat[~np.isnan(cat[z_keyname])]
    cat = cat[cat[z_keyname] < 1.3]
    cat_gal = cat[cat['CLASS'] < 10]
else:
    raise ValueError(cat_type+': cat type not acceptable')

cat_gal = cat_gal[cat_gal[mass_keyname] > 9.0]
print('cat_type', cat_type, ', param_type', params, z_keyname, mass_keyname, ', massive_selection', massive_selection)
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
cat_random = Table.read('s16a_' + cat_name + '_random.fits')
try:
    cat_random = cat_random[cat_random['inside_u'] == True]
except KeyError:
    cat_random = cat_random[cat_random['inside_uS'] == True]
cat_random_nomask = np.copy(cat_random)
cat_random = cat_random[cat_random['MASK'] == False]
cat_random_points = Table(names=('RA', 'DEC', 'GAL_ID'))   # to store position of selected random apertures
if cat_type == 'old' and cat_name == 'SXDS3_uddd':
    cat_name = 'SXDS_uddd'

# ############ main loop ################
for z in z_bins:
    z = round(z, 1)
    z_bin_size = 0.3 if all_z else 0.1

    # prepare directory for satellite catalogs
    sat_cat_dir = path + cat_name+'_'+str(z*10)+'/'
    if not os.path.exists(sat_cat_dir):
        os.system('mkdir '+sat_cat_dir)

    print('=============' + str(round(z-z_bin_size, 1)) + '<z<'+str(round(z+z_bin_size, 1))+'================')
    print(mode, csfq, ssfq_series, masscut_low, masscut_high, masscut_low_host, 'evo_masscut =',evo_masscut)
    cat_random_copy = np.copy(cat_random)  # reset random points catalog at each redshift

    # # use matched stellar-mass central sample?
    # cat_iso_cen_saved = Table.read('isolated_'+str(csfq)+'_central_matched_'+str(z)+'.fits')
    # iso_cen_saved_ids = cat_iso_cen_saved[id_keyname]

    # select centrals that satisfy both old and new selection
    if massive_selection == 'normal':
        cat_massive_gal = cat_gal[np.logical_and(cat_gal[mass_keyname] > masscut_low_host, cat_gal[mass_keyname] < masscut_high_host)]
        cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal[z_keyname] - z) < z_bin_size]
    else:
        cat_massive_z_slice = Table(names=cat_gal.colnames, dtype=[str(y[0]) for x, y in cat_gal.dtype.fields.items()])

        cat_massive_gal_old = cat_gal[np.logical_and(cat_gal['MASS_MED'] > 11.15, cat_gal['MASS_MED'] < 13.0)]
        cat_massive_z_slice_old = cat_massive_gal_old[abs(cat_massive_gal_old['zKDEPeak'] - z) < z_bin_size]
        cat_massive_gal_new = cat_gal[np.logical_and(cat_gal['MASS_MED_new'] > 11.35, cat_gal['MASS_MED_new'] < 13.0)]
        cat_massive_z_slice_new = cat_massive_gal_new[abs(cat_massive_gal_new['Z_BEST_BC03'] - z) < z_bin_size]
        if massive_selection == 'both':
            for gal in cat_massive_z_slice_old:
                if gal[id_keyname] in cat_massive_z_slice_new[id_keyname]:
                    cat_massive_z_slice.add_row(gal)
        elif massive_selection == 'new':
            for gal in cat_massive_z_slice_new:
                if gal[id_keyname] not in cat_massive_z_slice_old[id_keyname]:
                    cat_massive_z_slice.add_row(gal)
        elif massive_selection == 'old':
            for gal in cat_massive_z_slice_old:
                if gal[id_keyname] not in cat_massive_z_slice_new[id_keyname]:
                    cat_massive_z_slice.add_row(gal)
        elif massive_selection == 'both-central':
            cat_massive_both_central = Table.read('massive_gal_matched_cat/isolated_'+cat_name+'_'+str(z)+'_both_central.positions.fits')
            for gal in cat_massive_z_slice_old:
                if gal[id_keyname] in cat_massive_both_central[id_keyname]:
                    cat_massive_z_slice.add_row(gal)
        else:
            raise NameError('Not acceptable massive_selection param.')

    # ---------------------
    print('No. of massive gals:', len(cat_massive_z_slice))
    coord_massive_gal = SkyCoord(cat_massive_z_slice['RA'] * u.deg, cat_massive_z_slice['DEC'] * u.deg)
    cat_all_z_slice = cat_gal[abs(cat_gal[z_keyname] - z) < 0.4]  # initial slice to reduce computation

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
    companion_count_stack = 0
    n_sat = []  # number of satellites per central
    n_bkg = []  # number of bkg contamination per central
    R_m_stack = np.ones(bin_number) * 1e-6  # number of random points (masked) in each bin
    R_u_stack = np.ones(bin_number) * 1e-6  # number of random points (unmasked) in each bin
    R_m_stack_bkg = np.ones(bin_number) * 1e-6  # number of random points (masked, bkg apertures) in each bin
    R_u_stack_bkg = np.ones(bin_number) * 1e-6  # number of random points (unmasked, bkg apertures) in each bin

    # loop for all massive galaxies (potential central galaxy candidate)
    isolated_counts2 = 0
    isolated_cat = Table(names=cat_gal.colnames, dtype=[str(y[0]) for x, y in cat_gal.dtype.fields.items()])  # catalog of isolated central galaxies
    for gal in cat_massive_z_slice:
        massive_count += 1

        ### ONLY FOR MASS-MATCHED CENTRAL (split sfq) SAMPLE
        # if gal[id_keyname] not in iso_cen_saved_ids:
        #     isolated_counts -= 1
        #     continue

        print('Progress:' + str(massive_count) + '/' + str(len(cat_massive_z_slice)), end='\r')
        dis = WMAP9.angular_diameter_distance(gal[z_keyname]).value
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)
        if check_edge(gal['RA'], gal['DEC'], dis):
            continue

        # prepare neighbors catalog
        cat_neighbors_z_slice = cat_all_z_slice[abs(cat_all_z_slice[z_keyname] - gal[z_keyname]) < sat_z_cut * 0.044 * (1 + gal[z_keyname])]

        # (initial rectengular spatial cut)
        cat_neighbors = cat_neighbors_z_slice[abs(cat_neighbors_z_slice['RA'] - gal['RA']) < 2.5 / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[abs(cat_neighbors['DEC'] - gal['DEC']) < 2.5 / dis / np.pi * 180]

        # circular aperture cut, skip centrals that have no satellites
        if len(cat_neighbors) == 0:
            print('No Satellite for', gal[id_keyname])
            continue
        else:
            coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
            cat_neighbors = cat_neighbors[coord_neighbors.separation(coord_gal).degree < 0.7 / dis / np.pi * 180]

            cat_neighbors = cat_neighbors[cat_neighbors[id_keyname] != gal[id_keyname]]  # exclude central galaxy from satellite catalog
            if len(cat_neighbors) == 0:
                print('No Satellite for', gal[id_keyname])
                continue

        # isolation cut on central
        if gal[mass_keyname] < np.log10(isolation_factor) + max(cat_neighbors[mass_keyname]):  # no more-massive companions
            continue

        # TEMPORARY!
        # coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
        # cat_neighbors = cat_neighbors[coord_neighbors.separation(coord_gal).degree < 0.7 / dis / np.pi * 180]
        # if len(cat_neighbors) == 0:
        #     print('No Satellite for', gal[id_keyname])
        #     continue

        # add to the central catalog
        if not evo_masscut:  # if constant mass cut
            isolated_cat.add_row(gal)
        elif gal[mass_keyname] < 11.23 - 0.16 * gal[z_keyname]:  # if set evolving masscut, apply additional mass cut
            continue
        else:
            isolated_cat.add_row(gal)

        # cut on central SF/Q
        if csfq == 'csf' and gal['sfProb'] < sfprob_cut_high:
            continue
        elif csfq == 'cq' and gal['sfProb'] >= sfprob_cut_low:
            continue

        # cut on satellite sample (mass cut, ssfr cut)
        if isinstance(masscut_low, float):  # fixed cut
            cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] > masscut_low]
        elif isinstance(masscut_low, str):  # relative cut
            cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] > gal[mass_keyname]+np.log10(eval(masscut_low))]

        if isinstance(masscut_high, float):  # fixed cut
            cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] < masscut_high]
        elif isinstance(masscut_high, str):  # relative cut
            cat_neighbors = cat_neighbors[cat_neighbors[mass_keyname] < gal[mass_keyname]+np.log10(eval(masscut_high))]
        if len(cat_neighbors) == 0:
            if masscut_low_host == 11.15:
                isolated_cat.remove_row(-1)
            continue  # central gals which has no satellite

        # save satellite catalog (no bkg subtraction)
        mass_neighbors = cat_neighbors[mass_keyname]
        coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
        radius_list = coord_neighbors.separation(coord_gal).degree / 180. * np.pi * dis * 1000  # kpc
        if ('all' in ssfq_series) and save_catalogs:
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
        isolated_counts2 += 1
        sfq_weights = np.ones(len(radius_list))
        sfq_weights_ssf = cat_neighbors['sfProb']
        sfq_weights_sq = 1 - cat_neighbors['sfProb']

        count_binned = np.histogram(radius_list, weights=np.array(sfq_weights), bins=bin_edges)[0]
        count_binned_ssf = np.histogram(radius_list, weights=np.array(sfq_weights_ssf), bins=bin_edges)[0]
        count_binned_sq = np.histogram(radius_list, weights=np.array(sfq_weights_sq), bins=bin_edges)[0]
        sat_counts = np.array(count_binned, dtype='f8')
        sat_counts_ssf = np.array(count_binned_ssf, dtype='f8')
        sat_counts_sq = np.array(count_binned_sq, dtype='f8')

        # mask correction
        R_m, R_u = correct_for_masked_area(gal['RA'], gal['DEC'])
        R_m_stack += R_m
        R_u_stack += R_u

        # detection completeness correction
        z_three_bins = [0.4, 0.6, 0.8]
        z_corr = 0.4
        for z_corr_test in z_three_bins:
            if abs(gal[z_keyname] - z_corr_test) < abs(gal[z_keyname] - z_corr):
                z_corr = z_corr_test
        spatial_weight, spatial_weight_err = spatial_comp(z_corr, masscut_low, masscut_high, 'all')
        spatial_weight_ssf, spatial_weight_err_ssf = spatial_comp(z_corr, masscut_low, masscut_high, 'ssf')
        spatial_weight_sq, spatial_weight_err_sq = spatial_comp(z_corr, masscut_low, masscut_high, 'sq')
        radial_dist += sat_counts * spatial_weight
        radial_dist_ssf += sat_counts_ssf * spatial_weight
        radial_dist_sq += sat_counts_sq * spatial_weight

        # bkg removal
        if bkg_option == 'random':
            coord_random_list, sat_counts_bkg, sat_counts_bkg_ssf, sat_counts_bkg_sq, flag_bkg = bkg(cat_neighbors_z_slice, coord_massive_gal, gal[mass_keyname], dis)
            if flag_bkg == 0:
                radial_dist_bkg += sat_counts_bkg * np.average(spatial_weight[-5:])  # bkg completeness correction
                radial_dist_bkg_ssf += sat_counts_bkg_ssf * np.average(spatial_weight_ssf[-5:])
                radial_dist_bkg_sq += sat_counts_bkg_sq * np.average(spatial_weight_sq[-5:])
                bkg_count = bkg_count + 1
        else:
            sat_counts_bkg, sat_counts_bkg_ssf, sat_counts_bkg_sq = bkg_clustering(cat_neighbors_z_slice, gal['RA'], gal['DEC'], coord_gal, dis)
            radial_dist_bkg += sat_counts_bkg * np.ones(bin_number) * areas/area_c * np.average(spatial_weight[-5:])  # bkg completeness correction
            radial_dist_bkg_ssf += sat_counts_bkg_ssf * areas/area_c * np.average(spatial_weight_ssf[-5:])
            radial_dist_bkg_sq += sat_counts_bkg_sq * areas/area_c * np.average(spatial_weight_sq[-5:])
            bkg_count = bkg_count + 1

        # keep record of how many satellites a central has
        n_sat.append(sum(sat_counts))

    # ######### Aggregation of Results ##################
    # output central catalog
    z_output = 'allz' if all_z else str(z)
    if csfq == 'all' and ('all' in ssfq_series):
        n_sat_col = Column(data=n_sat, name='n_sat', dtype='i4')
        n_bkg_col = Column(data=np.ones(len(n_sat))*sum(radial_dist_bkg)/float(isolated_counts2), name='n_bkg', dtype='i4')
        if save_massive_catalogs:
            isolated_cat.add_columns([n_sat_col, n_bkg_col])  # n_sat and n_bkg are not corrected
            isolated_cat.write(sat_cat_dir+'isolated_'+cat_name+'_'+str(masscut_low_host)+'_'+z_output+'_massive_'+massive_selection+'_params_'+params+'.positions.fits', overwrite=True)

    print('Finished gals: ' + str(isolated_counts2) + '/' + str(len(cat_massive_z_slice)))
    print('Finished bkgs: ' + str(bkg_count))
    cat_random_points.write('random_points_' + cat_name + '_' + str(z) + '.fits', overwrite=True)
    radial_dist_series = [radial_dist, radial_dist_ssf, radial_dist_sq]
    radial_dist_bkg_series = [radial_dist_bkg, radial_dist_bkg_ssf, radial_dist_bkg_sq]
    for i, ssfq in enumerate(ssfq_series):
        # normalize number counts to by counting area and number of centrals
        radial_dist_norm = radial_dist_series[i] * R_u_stack / R_m_stack / float(isolated_counts2) / areas
        radial_dist_bkg_norm = radial_dist_bkg_series[i] * R_u_stack_bkg / R_m_stack_bkg / float(bkg_count) / areas

        # error estimation (assuming Poisson errors)
        spatial_weight, spatial_weight_err = spatial_comp(z, masscut_low, masscut_high, ssfq)
        err = cal_error(radial_dist_series[i], radial_dist_bkg_series[i], isolated_counts2, spatial_weight, spatial_weight_err) / areas
        n_central, n_count, n_err = isolated_counts2, radial_dist_norm - radial_dist_bkg_norm, err
        result = np.append([n_central], [n_count, n_err])

        # sat/bkg number density errors and results
        n_sat_err = cal_error2(radial_dist, isolated_counts2, spatial_weight, spatial_weight_err) / areas
        n_bkg_err = cal_error2(radial_dist_bkg, bkg_count, spatial_weight, spatial_weight_err) / areas
        result_sat = np.append([n_central], [radial_dist_norm, n_sat_err])
        result_bkg = np.append([n_central], [radial_dist_bkg_norm, n_bkg_err])

        # output result to file
        prefix = '_host_'+str(masscut_low_host) if split_central_mass else ''
        filename = path + str(mode) + cat_name + prefix + '_' + str(masscut_low) + '_' + str(csfq) + '_'  + str(ssfq) + '_' + z_output +'.txt'
        filename_sat = path + str(mode) + cat_name + prefix + '_sat_' + str(masscut_low) + '_' + str(csfq) + '_' + str(ssfq) + '_' + z_output + '.txt'
        filename_bkg = path + str(mode) + cat_name + prefix + '_bkg_' + str(masscut_low) + '_' + str(csfq) + '_' + str(ssfq) + '_' + z_output + '.txt'
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

