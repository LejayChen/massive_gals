import sys, os
from random import random
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.cosmology import WMAP9
from astropy.table import *
from mpi4py import MPI
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

# ################# START #####################
split_central_mass = False
all_z = False
correct_completeness = False
correct_masked = False
check_edge_flag = True
save_results = False
save_catalogs = False
evo_masscut = False
save_massive_catalogs = False

cen_selection = 'normal'  # pre_select or normal
cat_name = sys.argv[1]  # COSMOS_deep ELAIS_deep XMM-LSS_deep DEEP_deep SXDS_uddd
masscut_low_host = float(sys.argv[2])
masscut_high_host = float(sys.argv[3])
csfq = sys.argv[4]  # csf, cq, all
sat_z_cut = 3.0
masscut_low = 9.5
masscut_high = 12.0
sfq_keyname = 'sfProb_nuvrk'
rmax = 'fixed'
ssfq_series = ['all', 'ssf', 'sq']
z_bins = [0.6] if all_z else [0.4, 0.6]
bin_number = 14

# setting binning scheme
areas = np.array([])
radius_max = 0.7
bin_edges = 10 ** np.linspace(np.log10(10), np.log10(radius_max*1000), num=bin_number + 1)
for i in range(len(bin_edges[:-1])):
    areas = np.append(areas, (bin_edges[i + 1] ** 2 - bin_edges[i] ** 2) * np.pi)

cat = Table.read('/home/lejay/catalogs/v8_cats/' + cat_name + '_v8_gal_cut_params_sfq_added.fits')  # galaxy selection already done?
cat = cat[cat['MASK'] == 0]  # unmasked
cat_gal = cat[cat['OBJ_TYPE'] == 0]  # galaxies
z_keyname = 'ZPHOT'
mass_keyname = 'MASS_MED'
id_keyname = 'ID'
ra_key = 'RA'
dec_key = 'DEC'

if cat_name == 'XMM-LSS_deep':
    cat = cat[cat['inside_uS'] == True]
else:
    cat = cat[cat['inside_u'] == True]

cat_gal = cat_gal[~np.isnan(cat_gal[z_keyname])]
cat_gal = cat_gal[~np.isnan(cat_gal[mass_keyname])]
cat_gal = cat_gal[cat_gal[mass_keyname] > 9.0]
cat_gal = cat_gal[cat_gal['r'] < 25]
cat_gal = cat_gal[cat_gal[sfq_keyname] >= 0]
cat_gal = cat_gal[cat_gal[sfq_keyname] <= 1]

# output run settings and check save path
print('will NOT save results!')

# read-in random point catalog
cat_random = Table.read('/home/lejay/random_point_cat/'+cat_name + '_random_point.fits')
cat_random = cat_random[cat_random['inside'] != 0]
cat_random_nomask = np.copy(cat_random)
cat_random = cat_random[cat_random['MASK'] == 0]

# ############ main loop ################
cen_cat_dir = 'central_cat/'
print('############ ',  masscut_low_host, masscut_high_host)

for z_bin_count, z in enumerate(z_bins):
    z = round(z, 1)
    z_bin_size = 0.3 if all_z else 0.1
    cat_all_z_slice = cat_gal[cat_gal[z_keyname] < 5]
    cat_random_copy = np.copy(cat_random)  # reset random points catalog at each redshift

    # prepare directory for satellite catalogs
    print('=============' + str(round(z-z_bin_size, 1)) + '<z<'+str(round(z+z_bin_size, 1))+'================')
    print(csfq, ssfq_series, masscut_low, masscut_high, masscut_low_host, 'evo_masscut =', evo_masscut)
    cat_massive_gal = cat_gal[np.logical_and(cat_gal[mass_keyname] > masscut_low_host, cat_gal[mass_keyname] < masscut_high_host)]
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal[z_keyname] - z) < z_bin_size]
    coord_massive_gal = SkyCoord(np.array(cat_massive_z_slice[ra_key]) * u.deg, np.array(cat_massive_z_slice[dec_key]) * u.deg)
    print('No. of massive gals:', len(cat_massive_z_slice))

    # loop for all massive galaxies (potential central galaxy candidate)
    massive_count = 0
    isolated_counts = 0
    sf_count = 0
    q_count = 0
    isolated_cat = Table(names=cat_massive_z_slice.colnames, dtype=[str(y[0]) for x, y in cat_massive_z_slice.dtype.fields.items()])  # catalog of isolated central galaxies
    my_cat_massive_z_slice = cat_massive_z_slice

    for gal in my_cat_massive_z_slice:
        massive_count += 1
        isolation_factor = 10 ** 0
        dis = WMAP9.angular_diameter_distance(gal[z_keyname]).value
        coord_gal = SkyCoord(gal[ra_key] * u.deg, gal[dec_key] * u.deg)
        if check_edge_flag and check_edge(gal[ra_key], gal[dec_key], dis):
            continue

        # prepare satellites catalog
        cat_neighbors_z_slice = cat_all_z_slice[abs(cat_all_z_slice[z_keyname] - gal[z_keyname]) < sat_z_cut * 0.044 * (1 + gal[z_keyname])]
        cat_neighbors = cat_neighbors_z_slice[abs(cat_neighbors_z_slice[ra_key] - gal[ra_key]) < radius_max * 4 / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[abs(cat_neighbors[dec_key] - gal[dec_key]) < radius_max * 4 / dis / np.pi * 180] # circular aperture cut
        if len(cat_neighbors) == 0:  # skip centrals that have no satellites
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
                continue

        # isolation cut on central
        if gal[mass_keyname] < np.log10(isolation_factor) + max(cat_neighbors[mass_keyname]):  # no more-massive companions
            continue

        # evo mass cut
        if evo_masscut and gal[mass_keyname] < 11.4 - 0.16 * gal[z_keyname]:  # if set evolving masscut, apply additional mass cut
            continue

        # cut on central SF/Q
        isolated_counts += 1
        if gal[sfq_keyname] > 0.5:  # Q
            sf_count += 1
        elif gal[sfq_keyname] <= 0.5:  # SF
            q_count += 1

    # ######### Collect/Write Results ##################
    # print out central statistics info
    print(cat_name, z, isolated_counts, sf_count, q_count)
