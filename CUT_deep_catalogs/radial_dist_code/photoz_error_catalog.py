import sys
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import *
from mpi4py import MPI

comm=MPI.COMM_WORLD
rank = comm.Get_rank()
nProcs = comm.Get_size()

cat_names = ['COSMOS_deep', 'XMM-LSS_deep']
cat_name = cat_names[rank]
sample_selection = sys.argv[1]
pair_sfq = sys.argv[4]
catalog_type = sys.argv[6]  # lephare or phos
zkeyname = sys.argv[7]

if catalog_type == 'lephare':
    # v9 catalog
    cat = Table.read('/home/lejay/catalogs/v9_cats/' + cat_name + '_v9_gal_cut_params_sfq_added.fits')  # galaxy selection already done?
    if cat_name == 'XMM-LSS_deep':
        cat = cat[cat['inside_uS'] == True]
    else:
        cat = cat[cat['inside_u'] == True]
    cat = cat[cat['MASK'] == 0]  # unmasked
    cat_gal = cat[cat['OBJ_TYPE'] == 0]  # galaxies
else:
    # phosphoros catalog
    cat = Table.read('/home/lejay/catalogs/p_cats/' + cat_name + '_pcat_211221_gal_cut_params.fits')  # galaxy selection already done?
    if cat_name == 'XMM-LSS_deep':
        cat = cat[cat['inside_uS'] == True]
    else:
        cat = cat[cat['inside_u'] == True]
    cat = cat[cat['inside_hsc'] == True]
    cat = cat[cat['isOutsideMask'] == 1]  # unmasked
    cat_gal = cat[cat['isStar'] == 0]  # galaxies
    cat_gal = cat_gal[cat_gal['ZPHOT_NIR'] > -90]  # only objects with NIR derived redshifts

cat_gal = cat_gal[cat_gal[zkeyname] > 0]  # z cut
cat_gal = cat_gal[cat_gal[zkeyname] < 7.0]  # z cut
ra_name = 'RA'
dec_name = 'DEC'
mag_bright = eval(sys.argv[2])
mag_faint = eval(sys.argv[3])

if sample_selection == 'mag':
    cat_gal = cat_gal[cat_gal['i'] > mag_bright]
    cat_gal = cat_gal[cat_gal['i'] < mag_faint]
    z_low = 0.0
    z_high = 7.0
elif sample_selection == 'magz':
    cat_gal = cat_gal[cat_gal['i'] > mag_bright]
    cat_gal = cat_gal[cat_gal['i'] < mag_faint]
    z_low = eval(sys.argv[5])-0.5
    z_high = eval(sys.argv[5])+0.5
elif sample_selection == 'z':
    cat_gal = cat_gal[cat_gal['i'] > mag_bright]
    cat_gal = cat_gal[cat_gal['i'] < mag_faint]
    z_low = eval(sys.argv[5])-0.5
    z_high = eval(sys.argv[5])+0.5
else:
    raise ValueError
print('selected in '+cat_name+':', len(cat_gal))

# load random catalog
cat_random = Table.read('/home/lejay/random_point_cat/'+cat_name + '_random_point.fits')
cat_random = cat_random[cat_random['inside'] == 0]
cat_random = cat_random[cat_random['MASK'] != 0]
bin_number = 350
aper_size = 150.0  # arcsec, aperture size
max_sep = 10.0  # arcsec, max separation for close pairs
number_pairs = 75000  # number of pairs

i = 0
fails = 0
number_pairs_count = 0
success = True
deltaz_close = np.zeros(bin_number)
deltaz_random = np.zeros(bin_number)
deltaz_physical = np.zeros(bin_number)

while number_pairs_count < number_pairs:
    if fails > number_pairs:
        success = False
        break

    # ############### (close pairs) random position ########### #
    deltaz_list = []
    rand_id = np.random.randint(len(cat_random))
    ra_rand = cat_random[rand_id]['RA']
    dec_rand = cat_random[rand_id]['DEC']
    coord_center = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)

    # circular aperture cut
    coord_gals = SkyCoord(np.array(cat_gal['RA']) * u.deg, np.array(cat_gal['DEC']) * u.deg)
    cat_neighbors = cat_gal[coord_gals.separation(coord_center).degree < aper_size / 3600.]

    # redshift range for gal1 in each pair
    cat_neighbors_z = cat_neighbors[cat_neighbors[zkeyname] > z_low]
    cat_neighbors_z = cat_neighbors_z[cat_neighbors_z[zkeyname] < z_high]

    # sf/q
    if pair_sfq == 'sf-sf':
        cat_neighbors_z = cat_neighbors_z[cat_neighbors_z['sfProb_nuvrk'] > 0.5]
        cat_neighbors = cat_neighbors[cat_neighbors['sfProb_nuvrk'] > 0.5]
    elif pair_sfq == 'sf-q':
        cat_neighbors_z = cat_neighbors_z[cat_neighbors_z['sfProb_nuvrk'] > 0.5]
        cat_neighbors = cat_neighbors[cat_neighbors['sfProb_nuvrk'] < 0.5]
    elif pair_sfq == 'q-sf':
        cat_neighbors_z = cat_neighbors_z[cat_neighbors_z['sfProb_nuvrk'] < 0.5]
        cat_neighbors = cat_neighbors[cat_neighbors['sfProb_nuvrk'] > 0.5]
    elif pair_sfq == 'q-q':
        cat_neighbors_z = cat_neighbors_z[cat_neighbors_z['sfProb_nuvrk'] < 0.5]
        cat_neighbors = cat_neighbors[cat_neighbors['sfProb_nuvrk'] < 0.5]

    coord_neighbors = SkyCoord(np.array(cat_neighbors['RA']) * u.deg, np.array(cat_neighbors['DEC']) * u.deg)   # coord of gal2's
    coord_gals_z = SkyCoord(np.array(cat_neighbors_z['RA']) * u.deg, np.array(cat_neighbors_z['DEC']) * u.deg)  # coord of gal1's
    if len(cat_neighbors_z) > 4:
        arr = search_around_sky(coord_gals_z, coord_neighbors, max_sep/3600 * u.deg)
        sep2d, b = np.unique(np.round(arr[2], 10), return_index=True)  # only keep the unique pairs
        arr0 = arr[0][b]
        arr1 = arr[1][b]

        # arr_all = search_around_sky(coord_gals_z, coord_neighbors, aper_size*2/3600 * u.deg)
        # a2, b2 = np.unique(np.round(arr_all[2], 10), return_index=True)
        # no_pairs_all = len(b2)

        gal1_list = arr0[sep2d.value > 1.5 / 3600]
        gal2_list = arr1[sep2d.value > 1.5 / 3600]
        no_pairs = len(gal1_list)

        if no_pairs == 0:
            fails += 1
            continue
        else:
            number_pairs_count += no_pairs

        # calculate delta_z
        for k in range(len(gal1_list)):
            gal1 = cat_neighbors_z[gal1_list[k]]
            gal2 = cat_neighbors[gal2_list[k]]
            deltaz = (gal1[zkeyname]-gal2[zkeyname]) / (1+(gal1[zkeyname]+gal2[zkeyname])/2)

            # append
            deltaz_list.append(deltaz)

        # Nd = len(cat_neigdhbors_z)
        # fg = no_pairs/no_pairs_all
        # fz = 2

    else:
        fails += 1
        continue

    # ################## (random pairs) random point #####
    deltaz_list_rand = []
    rand_id = np.random.randint(len(cat_random))
    ra_rand = cat_random[rand_id]['RA']
    dec_rand = cat_random[rand_id]['DEC']
    coord_center = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)

    # circular aperture cut
    coord_gals = SkyCoord(np.array(cat_random['RA']) * u.deg, np.array(cat_random['DEC']) * u.deg)
    cat_neighbors_rand = cat_random[coord_gals.separation(coord_center).degree < aper_size / 3600.]

    # randomly replace redshifts from data catalog
    cat_random_select = np.random.choice(cat_gal, size=len(cat_neighbors_rand))
    cat_neighbors_rand[zkeyname] = cat_random_select[zkeyname]
    if pair_sfq != 'all-all':
        cat_neighbors_rand['sfProb_nuvrk'] = cat_random_select['sfProb_nuvrk']

    # redshift range for gal1 in each pair
    cat_neighbors_z_rand = cat_neighbors_rand[cat_neighbors_rand[zkeyname] > z_low]
    cat_neighbors_z_rand = cat_neighbors_z_rand[cat_neighbors_z_rand[zkeyname] < z_high]

    # sf/q
    if pair_sfq == 'sf-sf':
        cat_neighbors_z_rand = cat_neighbors_z_rand[cat_neighbors_z_rand['sfProb_nuvrk'] > 0.5]
        cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['sfProb_nuvrk'] > 0.5]
    elif pair_sfq == 'sf-q':
        cat_neighbors_z_rand = cat_neighbors_z_rand[cat_neighbors_z_rand['sfProb_nuvrk'] > 0.5]
        cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['sfProb_nuvrk'] < 0.5]
    elif pair_sfq == 'q-sf':
        cat_neighbors_z_rand = cat_neighbors_z_rand[cat_neighbors_z_rand['sfProb_nuvrk'] < 0.5]
        cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['sfProb_nuvrk'] > 0.5]
    elif pair_sfq == 'q-q':
        cat_neighbors_z_rand = cat_neighbors_z_rand[cat_neighbors_z_rand['sfProb_nuvrk'] < 0.5]
        cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['sfProb_nuvrk'] < 0.5]

    coord_neighbors_rand = SkyCoord(np.array(cat_neighbors_rand['RA']) * u.deg, np.array(cat_neighbors_rand['DEC']) * u.deg)   # coord of gal2's
    coord_gals_z_rand = SkyCoord(np.array(cat_neighbors_z_rand['RA']) * u.deg, np.array(cat_neighbors_z_rand['DEC']) * u.deg)  # coord of gal1's
    if len(cat_neighbors_z_rand) > 4:
        arr = search_around_sky(coord_gals_z_rand, coord_neighbors_rand, aper_size*2 / 3600 * u.deg)
        sep2d, b = np.unique(np.round(arr[2], 10), return_index=True)  # only keep the unique pairs
        arr0 = arr[0][b]
        arr1 = arr[1][b]

        gal1_list = arr0[sep2d.value > 1.5 / 3600]
        gal2_list = arr1[sep2d.value > 1.5 / 3600]
        if len(gal1_list) == 0:
            fails += 1
            continue

        # calculate deltaz
        for k in range(len(gal1_list)):
            gal1 = cat_neighbors_z_rand[gal1_list[k]]
            gal2 = cat_neighbors_rand[gal2_list[k]]
            deltaz = (gal1[zkeyname] - gal2[zkeyname]) / (1 + (gal1[zkeyname] + gal2[zkeyname]) / 2)

            # append
            deltaz_list_rand.append(deltaz)
    else:
        fails += 1
        continue

    # calculate histogram
    deltaz_hist, bin_edges = np.histogram(deltaz_list, bins=bin_number, range=(-2, 2))
    deltaz_rand_hist, bin_edges2 = np.histogram(deltaz_list_rand, bins=bin_number, range=(-2, 2))

    # subtract histogram
    if len(deltaz_hist[np.isnan(deltaz_hist)]) == 0 and len(deltaz_hist[np.isnan(deltaz_rand_hist)]) == 0 and len(deltaz_list_rand) > 0:
        deltaz_close += deltaz_hist
        deltaz_random += deltaz_rand_hist * len(deltaz_list) / len(deltaz_list_rand)
        deltaz_physical += deltaz_hist - deltaz_rand_hist * len(deltaz_list) / len(deltaz_list_rand)
        i += 1

# save results
if success:
    print(cat_name, 'done')
    if pair_sfq == 'all-all':
        filename_base = 'delta_z/catalog_'+str(z_low)+'_'+str(z_high)+'_'+sample_selection+'_'+\
                str(mag_bright)+'_'+str(mag_faint)+'_'+cat_name+'_'+catalog_type+'_'+zkeyname
    else:
        filename_base = 'delta_z/catalog_'+str(z_low)+'_'+str(z_high)+'_'+sample_selection+'_'+\
                str(mag_bright)+'_'+str(mag_faint)+'_'+pair_sfq+'_'+cat_name+'_'+catalog_type+'_'+zkeyname
    np.save(filename_base+'_deltaz_close.npy', deltaz_close)
    np.save(filename_base+'_deltaz_random.npy', deltaz_random)
    np.save(filename_base+'_deltaz_physical.npy', deltaz_physical)
else:
    print(cat_name, 'failed')



