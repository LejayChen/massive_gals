import sys
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nProcs = comm.Get_size()

sample_selection = sys.argv[1]
pair_sfq = sys.argv[4]

# cosmos2020
cat = Table.read('/home/lejay/catalogs/photoz_cosmos2020_lephare_classic_v1.8_trim_Sfq_phot_Added.fits')
cat_gal = cat[cat['type'] == 0]  # galaxies
cat_gal = cat_gal[cat_gal['zPDF'] > 0]  # z cut
cat_gal = cat_gal[cat_gal['zPDF'] < 7.0]  # z cut

if sample_selection == 'mass':
    z_select = False
    # mass cut
    mass_low = eval(sys.argv[2])
    mass_high = eval(sys.argv[3])
    z_low = 0.0
    z_high = 7.0
    cat_gal = cat_gal[cat_gal['mass_med'] > mass_low]
    cat_gal = cat_gal[cat_gal['mass_med'] < mass_high]
    selection_low = mass_low
    selection_high = mass_high
elif sample_selection == 'mag':
    z_select = False
    # mag cut
    mag_bright = eval(sys.argv[2])
    mag_faint = eval(sys.argv[3])
    z_low = 0.0
    z_high = 7.0
    cat_gal = cat_gal[cat_gal['i'] > mag_bright]
    cat_gal = cat_gal[cat_gal['i'] < mag_faint]
    selection_low = mag_bright
    selection_high = mag_faint

elif sample_selection == 'z':
    z_select = True
    selection_low = 10
    selection_high = 26
    cat_gal = cat_gal[cat_gal['i'] > selection_low]
    cat_gal = cat_gal[cat_gal['i'] < selection_high]
    z_low = eval(sys.argv[2])
    z_high = eval(sys.argv[3])
else:
    raise ValueError

print(sample_selection, selection_low, selection_high)

i = 0
bin_number = 250
len_cat_nei = []
deltaz_close = np.zeros(bin_number)
deltaz_random = np.zeros(bin_number)
deltaz_physical = np.zeros(bin_number)
aper_size = 80.0  # arcsec
while i < 300:

    # (close pairs) random position
    deltaz_list = []
    rand_id = np.random.randint(len(cat_gal))
    ra_rand = cat_gal[rand_id]['ra']
    dec_rand = cat_gal[rand_id]['dec']
    coord_center = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)

    # circular aperture cut (60 arcsec)
    coord_gals = SkyCoord(np.array(cat_gal['ra']) * u.deg, np.array(cat_gal['dec']) * u.deg)
    cat_neighbors = cat_gal[coord_gals.separation(coord_center).degree < aper_size / 3600.]

    # redshift range for gal1 in each pair
    cat_neighbors_z = cat_neighbors[cat_neighbors['zPDF'] > z_low]
    cat_neighbors_z = cat_neighbors_z[cat_neighbors_z['zPDF'] < z_high]

    # sf/q
    if pair_sfq == 'sf-sf':
        cat_neighbors_z = cat_neighbors_z[cat_neighbors_z['sSFR_med'] > -11]
        cat_neighbors = cat_neighbors[cat_neighbors['sSFR_med'] > -11]
    elif pair_sfq == 'sf-q':
        cat_neighbors_z = cat_neighbors_z[cat_neighbors_z['sSFR_med'] > -11]
        cat_neighbors = cat_neighbors[cat_neighbors['sSFR_med'] < -11]
    elif pair_sfq == 'q-sf':
        cat_neighbors_z = cat_neighbors_z[cat_neighbors_z['sSFR_med'] < -11]
        cat_neighbors = cat_neighbors[cat_neighbors['sSFR_med'] > -11]
    elif pair_sfq == 'q-q':
        cat_neighbors_z = cat_neighbors_z[cat_neighbors_z['sSFR_med'] < -11]
        cat_neighbors = cat_neighbors[cat_neighbors['sSFR_med'] < -11]

    coord_neighbors = SkyCoord(np.array(cat_neighbors['ra']) * u.deg, np.array(cat_neighbors['dec']) * u.deg)
    coord_gals_z = SkyCoord(np.array(cat_neighbors_z['ra']) * u.deg, np.array(cat_neighbors_z['dec']) * u.deg)

    if len(cat_neighbors_z) > 5:
        len_cat_nei.append(len(cat_neighbors_z))

        arr = search_around_sky(coord_gals_z, coord_neighbors, 10. / 3600 * u.deg)
        arr_all = search_around_sky(coord_gals_z, coord_neighbors, aper_size * 2 / 3600 * u.deg)
        gal1_list = arr[0][arr[2].value > 2.5 / 3600]
        gal2_list = arr[1][arr[2].value > 2.5 / 3600]
        no_pairs = len(gal1_list)
        no_pairs_all = len(arr_all[0])
        if no_pairs > 0:
            print('no of objects', len(cat_neighbors_z), 'no. of close pairs', no_pairs, 'no. of pairs ', no_pairs_all)
        else:
            continue

        # calculate deltaz
        for k in range(len(gal1_list)):
            if gal1_list[k] < gal2_list[k]:  # only unique pairs
                gal1 = cat_neighbors_z[gal1_list[k]]
                gal2 = cat_neighbors[gal2_list[k]]
                deltaz = (gal1['zPDF'] - gal2['zPDF']) / (1 + (gal1['zPDF'] + gal2['zPDF']) / 2)

                # append
                deltaz_list.append(deltaz)
            else:
                continue

        Nd = len(cat_neighbors_z)
        fg = no_pairs / no_pairs_all
        fz = 2

    else:
        continue

    # (random pairs) random point
    deltaz_list_rand = []
    rand_id = np.random.randint(len(cat_gal))
    ra_rand = cat_gal[rand_id]['ra']
    dec_rand = cat_gal[rand_id]['dec']
    coord_center = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)

    # circular aperture cut (60 arcsec)
    coord_gals = SkyCoord(np.array(cat_gal['ra']) * u.deg, np.array(cat_gal['dec']) * u.deg)
    cat_neighbors_rand = cat_gal[coord_gals.separation(coord_center).degree < aper_size / 3600.]

    # randomly replace redshifts from other objects
    cat_random_select = np.random.choice(cat_gal, size=len(cat_neighbors_rand))
    cat_neighbors_rand['zPDF'] = cat_random_select['zPDF']
    cat_neighbors_rand['sSFR_med'] = cat_random_select['sSFR_med']

    # redshift range for gal1 in each pair
    cat_neighbors_z_rand = cat_neighbors_rand[cat_neighbors_rand['zPDF'] > z_low]
    cat_neighbors_z_rand = cat_neighbors_z_rand[cat_neighbors_z_rand['zPDF'] < z_high]

    # sf/q
    if pair_sfq == 'sf-sf':
        cat_neighbors_z_rand = cat_neighbors_z_rand[cat_neighbors_z_rand['sSFR_med'] > -11]
        cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['sSFR_med'] > -11]
    elif pair_sfq == 'sf-q':
        cat_neighbors_z_rand = cat_neighbors_z_rand[cat_neighbors_z_rand['sSFR_med'] > -11]
        cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['sSFR_med'] < -11]
    elif pair_sfq == 'q-sf':
        cat_neighbors_z_rand = cat_neighbors_z_rand[cat_neighbors_z_rand['sSFR_med'] < -11]
        cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['sSFR_med'] > -11]
    elif pair_sfq == 'q-q':
        cat_neighbors_z_rand = cat_neighbors_z_rand[cat_neighbors_z_rand['sSFR_med'] < -11]
        cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['sSFR_med'] < -11]

    coord_neighbors_rand = SkyCoord(np.array(cat_neighbors_rand['ra']) * u.deg,
                                    np.array(cat_neighbors_rand['dec']) * u.deg)
    coord_gals_z_rand = SkyCoord(np.array(cat_neighbors_z_rand['ra']) * u.deg,
                                 np.array(cat_neighbors_z_rand['dec']) * u.deg)

    if len(cat_neighbors_z_rand) > 5:
        arr = search_around_sky(coord_gals_z_rand, coord_neighbors_rand, aper_size * 2 / 3600 * u.deg)
        gal1_list = arr[0][arr[2].value > 2.5 / 3600]
        gal2_list = arr[1][arr[2].value > 2.5 / 3600]
        if len(gal1_list) > 0:
            print('no of objects', len(cat_neighbors_z), 'no. of close pairs', no_pairs, 'no. of pairs ', no_pairs_all)
        else:
            continue

        # calculate deltaz
        for k in range(len(gal1_list)):
            if gal1_list[k] < gal2_list[k]:  # only unique pairs
                gal1 = cat_neighbors_z_rand[gal1_list[k]]
                gal2 = cat_neighbors_rand[gal2_list[k]]
                deltaz = (gal1['zPDF'] - gal2['zPDF']) / (1 + (gal1['zPDF'] + gal2['zPDF']) / 2)

                # append
                deltaz_list_rand.append(deltaz)
            else:
                continue
    else:
        continue

    # calculate histogram
    deltaz_hist, bin_edges = np.histogram(deltaz_list, bins=bin_number, range=(-2, 2))
    deltaz_rand_hist, bin_edges2 = np.histogram(deltaz_list_rand, bins=bin_number, range=(-2, 2))

    # subtract histogram

    print('check:', i, len(deltaz_list), 0.5 * Nd * (Nd - 1) * fg * fz)
    if len(deltaz_hist[np.isnan(deltaz_hist)]) == 0 and len(deltaz_hist[np.isnan(deltaz_rand_hist)]) == 0 and len(
            deltaz_list_rand) > 0:
        deltaz_close += deltaz_hist
        deltaz_random += deltaz_rand_hist * len(deltaz_list) / len(deltaz_list_rand)
        deltaz_physical += deltaz_hist - deltaz_rand_hist * len(deltaz_list) / len(deltaz_list_rand)
        i += 1

len_cat_nei = np.array(len_cat_nei)

print(deltaz_physical)
# save results

if pair_sfq == 'all-all':
    filename_base = 'delta_z/catalog_' + str(z_low) + '_' + str(z_high) + '_' + sample_selection + '_' + \
                    str(selection_low) + '_' + str(selection_high) + '_cosmos2020'
else:
    filename_base = 'delta_z/cosmos2020_catalog_' + str(z_low) + '_' + str(z_high) + '_' + sample_selection + '_' + \
                    str(selection_low) + '_' + str(selection_high) + '_' + pair_sfq + '_cosmos2020'
np.save(filename_base + '_deltaz_close.npy', deltaz_close)
np.save(filename_base + '_deltaz_random.npy', deltaz_random)
np.save(filename_base + '_deltaz_physical.npy', deltaz_physical)

# np.save(filename_base+'_deltaz_rand.npy', deltaz_list_rand)


