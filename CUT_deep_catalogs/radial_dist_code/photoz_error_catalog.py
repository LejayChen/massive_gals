import sys
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import *
from mpi4py import MPI


comm=MPI.COMM_WORLD
rank = comm.Get_rank()
nProcs = comm.Get_size()
sample_selection = sys.argv[1]
mag_bright = eval(sys.argv[2])
mag_faint = eval(sys.argv[3])
pair_sfq = sys.argv[4]
catalog_type = sys.argv[6]  # lephare or phos
zkeyname = sys.argv[7]
cat_name = sys.argv[8]
save_catalog = True
ra_name = 'RA'
dec_name = 'DEC'

print('loading tables ...')
if catalog_type == 'lephare':
    cat = Table.read('/home/lejay/catalogs/v9_cats/' + cat_name + '_v9_gal_cut_params_sfq_added.fits')
    if cat_name == 'XMM-LSS_deep':
        cat = cat[cat['inside_uS'] == True]
    else:
        cat = cat[cat['inside_u'] == True]
    cat = cat[cat['MASK'] == 0]  # unmasked
    cat_gal = cat[cat['OBJ_TYPE'] == 0]  # galaxies
elif catalog_type == 'lephare-t':
    cat_gal = Table.read('/home/lejay/catalogs/l_cats/' + cat_name + '_v9_lephare_thibaud.fits')   # galaxy selection already done !
else:
    cat = Table.read('/home/lejay/catalogs/p_cats/' + cat_name + '_pcat_211221_gal_cut_params.fits')
    if cat_name == 'XMM-LSS_deep':
        cat = cat[cat['inside_uS'] == True]
    else:
        cat = cat[cat['inside_u'] == True]
    cat = cat[cat['inside_hsc'] == True]
    cat = cat[cat['isOutsideMask'] == 1]  # unmasked
    cat_gal = cat[cat['isStar'] == 0]  # galaxies

    if cat_name in ['XMM-LSS_deep','COSMOS_deep']:
        if zkeyname == 'ZPHOT_NIR' or zkeyname == 'ZPHOT_6B_NIRarea':
            cat_gal = cat_gal[cat_gal['ZPHOT_NIR'] > -90]  # only objects with NIR derived redshifts
    else:
        zkeyname = 'ZPHOT'

if zkeyname == 'ZPHOT_6B_NIRarea':
    zkeyname = 'ZPHOT_6B'
    zkeyname_output = 'ZPHOT_6B_NIRarea'
else:
    zkeyname_output = zkeyname

cat_gal = cat_gal[cat_gal[zkeyname] > 0]  # z cut
cat_gal = cat_gal[cat_gal[zkeyname] < 7.0]  # z cut
if sample_selection == 'mag':
    cat_gal = cat_gal[cat_gal['i'] > mag_bright]
    cat_gal = cat_gal[cat_gal['i'] < mag_faint]
    z_low = 0.0
    z_high = 7.0
elif sample_selection == 'magz':
    cat_gal = cat_gal[cat_gal['i'] > mag_bright]
    cat_gal = cat_gal[cat_gal['i'] < mag_faint]
    z_low = eval(sys.argv[5])-0.05
    z_high = eval(sys.argv[5])+0.05
elif sample_selection == 'z':
    cat_gal = cat_gal[cat_gal['i'] > mag_bright]
    cat_gal = cat_gal[cat_gal['i'] < mag_faint]
    z_low = eval(sys.argv[5])-0.05
    z_high = eval(sys.argv[5])+0.05
else:
    raise ValueError

# load random catalog
cat_random = Table.read('/home/lejay/random_point_cat/'+cat_name + '_random_point.fits')
cat_random = cat_random[cat_random['inside'] == 0]
cat_random = cat_random[cat_random['MASK'] != 0]

# aperture settings
aper_size = 150.0  # arcsec, aperture size
max_sep = 10.0  # arcsec, max separation for close pairs
rand_close_ratio = 5

base = 100
if mag_faint < 23:
    number_pairs = 5*base
    bin_number = 220
elif mag_faint < 25:
    number_pairs = 10*base
    bin_number = 170
elif mag_faint < 26:
    number_pairs = 40*base
    bin_number = 130
else:
    number_pairs = 130*base
    bin_number = 120

i = 0
fails = 0
number_pairs_count = 0
success = True
deltaz_close = np.zeros(bin_number)
deltaz_random = np.zeros(bin_number)
deltaz_physical = np.zeros(bin_number)
deltaz_list_tot = []
deltaz_list_rand_tot = []
if cat_name in ['COSMOS_deep', 'XMM-LSS_deep']:
    cat_pair_col_names = ['ID1','RA1','DEC1','z1_NIR','z1_6B','imag1','ID2','RA2','DEC2','z2_NIR','z2_6B','imag2','dz_1pz']
    cat_pair_col_dtypes = ['i8','f8','f8','f8','f8','f8','i8','f8','f8','f8','f8','f8','f8']
    cat_pair_col_names_rand = ['RA1','DEC1','z1_NIR','z1_6B','RA2','DEC2','z2_NIR','z2_6B','dz_1pz']
    cat_pair_col_dtypes_rand = ['f8','f8','f8','f8','f8','f8','f8','f8','f8']
else:
    cat_pair_col_names = ['ID1','RA1','DEC1','z1','imag1','ID2','RA2','DEC2','z2','imag2','dz_1pz']
    cat_pair_col_dtypes = ['i8','f8','f8','f8','f8','i8','f8','f8','f8','f8','f8']
    cat_pair_col_names_rand = ['RA1','DEC1','z1','RA2','DEC2','z2','dz_1pz']
    cat_pair_col_dtypes_rand = ['f8','f8','f8','f8','f8','f8','f8']
cat_pair_close = Table(names=cat_pair_col_names, dtype=cat_pair_col_dtypes)
cat_pair_random = Table(names=cat_pair_col_names_rand, dtype=cat_pair_col_dtypes_rand)
while number_pairs_count < number_pairs:
    # print(mag_bright, mag_faint, cat_name, 'number_pairs_count', number_pairs_count)
    if fails > number_pairs:
        success = False
        break

    # ############### (close pairs) random position ########### #
    deltaz_list = []
    rand_id = np.random.randint(len(cat_random))
    ra_rand = cat_random[rand_id]['RA']
    dec_rand = cat_random[rand_id]['DEC']
    coord_center = SkyCoord(ra_rand * u.deg, dec_rand * u.deg)

    # circular aperture cut (aper_size)
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
    if len(cat_neighbors_z) > 3:
        arr = search_around_sky(coord_gals_z, coord_neighbors, max_sep/3600 * u.deg)
        sep2d, b = np.unique(np.round(arr[2], 10), return_index=True)  # only keep the unique pairs
        arr0 = arr[0][b]
        arr1 = arr[1][b]

        # arr_all = search_around_sky(coord_gals_z, coord_neighbors, aper_size*2/3600 * u.deg)
        # a2, b2 = np.unique(np.round(arr_all[2], 10), return_index=True)
        # no_pairs_all = len(b2)

        gal1_list = arr0[sep2d.value > 2.5 / 3600]
        gal2_list = arr1[sep2d.value > 2.5 / 3600]
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

            rand_num = np.random.randint(2)
            if rand_num == 1:
                gal_temp = gal1
                gal1 = gal2
                gal2 = gal_temp

            # append
            deltaz_list.append(deltaz)
            if cat_name in ['COSMOS_deep','XMM-LSS_deep']:
                cat_pair_close.add_row([gal1['ID'], gal1['RA'], gal1['DEC'], gal1['ZPHOT_NIR'],gal1['ZPHOT_6B'], gal1['i'], gal2['ID'], gal2['RA'], gal2['DEC'], gal2['ZPHOT_NIR'],gal2['ZPHOT_6B'], gal2['i'], deltaz])
            else:
                cat_pair_close.add_row([gal1['ID'], gal1['RA'], gal1['DEC'], gal1[zkeyname], gal1['i'], gal2['ID'], gal2['RA'],gal2['DEC'], gal2[zkeyname], gal2['i'], deltaz])

        # Nd = len(cat_neigdhbors_z)
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
    if cat_name in ['COSMOS_deep', 'XMM-LSS_deep']:
        cat_neighbors_rand['ZPHOT_NIR'] = cat_random_select['ZPHOT_NIR']
        cat_neighbors_rand['ZPHOT_6B'] = cat_random_select['ZPHOT_6B']
    else:
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
    if len(cat_neighbors_z_rand) > 3:
        arr = search_around_sky(coord_gals_z_rand, coord_neighbors_rand, aper_size*2 / 3600 * u.deg)
        sep2d, b = np.unique(np.round(arr[2], 10), return_index=True)  # only keep the unique pairs
        arr0 = arr[0][b]
        arr1 = arr[1][b]

        gal1_list = arr0[sep2d.value > 2.5 / 3600]
        gal2_list = arr1[sep2d.value > 2.5 / 3600]
        if len(gal1_list) == 0:
            fails += 1
            continue

        # calculate deltaz
        for k in range(len(gal1_list)):
            gal1 = cat_neighbors_z_rand[gal1_list[k]]
            gal2 = cat_neighbors_rand[gal2_list[k]]

            rand_num = np.random.randint(2)
            if rand_num == 1:
                gal_temp = gal1
                gal1 = gal2
                gal2 = gal_temp

            deltaz = (gal1[zkeyname] - gal2[zkeyname]) / (1 + (gal1[zkeyname] + gal2[zkeyname]) / 2)

            # append
            deltaz_list_rand.append(deltaz)
            if len(cat_pair_random) < rand_close_ratio * len(deltaz_list):
                if cat_name in ['COSMOS_deep','XMM-LSS_deep']:
                    cat_pair_random.add_row([gal1['RA'], gal1['DEC'], gal1['ZPHOT_NIR'],gal1['ZPHOT_6B'], gal2['RA'], gal2['DEC'], gal2['ZPHOT_NIR'], gal2['ZPHOT_6B'], deltaz])
                else:
                    cat_pair_random.add_row([gal1['RA'], gal1['DEC'], gal1[zkeyname], gal2['RA'], gal2['DEC'], gal2[zkeyname], deltaz])
    else:
        fails += 1
        continue

    # shorten length of delta_z_list_random
    deltaz_list_rand = np.array(deltaz_list_rand)
    if len(deltaz_list_rand) > rand_close_ratio * len(deltaz_list):
        deltaz_list_rand = np.random.choice(deltaz_list_rand, size=rand_close_ratio * len(deltaz_list), replace=False)

    # append deltaz_list_tot
    deltaz_list_tot += deltaz_list
    deltaz_list_rand_tot += deltaz_list_rand.tolist()

    # calculate histogram
    deltaz_hist, bin_edges = np.histogram(deltaz_list, bins=bin_number, range=(-1, 1))
    deltaz_rand_hist, bin_edges2 = np.histogram(deltaz_list_rand, bins=bin_number, range=(-1, 1))

    # subtract histogram
    if len(deltaz_hist[np.isnan(deltaz_hist)]) == 0 and len(deltaz_hist[np.isnan(deltaz_rand_hist)]) == 0 and len(deltaz_list_rand) > 0:
        deltaz_close += deltaz_hist
        deltaz_random += deltaz_rand_hist * len(deltaz_list) / len(deltaz_list_rand)
        deltaz_physical += deltaz_hist - deltaz_rand_hist * len(deltaz_list) / len(deltaz_list_rand)
        i += 1

# save results
dir_list = 'list_delta_z/'
dir_hist = 'delta_z/'
dir_table = '/scratch/lejay/cat_delta_z/'
if success:
    if pair_sfq == 'all-all':
        filename_base = 'catalog_'+str(round(z_low,1))+'_'+str(round(z_high, 1))+'_'+sample_selection+'_'+\
                str(mag_bright)+'_'+str(mag_faint)+'_'+cat_name+'_'+catalog_type+'_'+zkeyname_output+'_'+str(rank)
    else:
        filename_base = 'catalog_'+str(round(z_low,1))+'_'+str(round(z_high, 1))+'_'+sample_selection+'_'+\
                str(mag_bright)+'_'+str(mag_faint)+'_'+pair_sfq+'_'+cat_name+'_'+catalog_type+'_'+zkeyname_output+'_'+str(rank)

    # save delta_z lists (no binning)
    # print('no. of close pairs:', len(np.array(deltaz_list_tot)))
    # print('no. of random pairs:', len(np.array(deltaz_list_rand_tot)))
    np.save(dir_list + filename_base + '_deltaz_list_close.npy', np.array(deltaz_list_tot))
    np.save(dir_list + filename_base + '_deltaz_list_random.npy', np.array(deltaz_list_rand_tot))

    # save histograms
    np.save(dir_hist + filename_base+'_deltaz_close.npy', deltaz_close)
    np.save(dir_hist + filename_base+'_deltaz_random.npy', deltaz_random)
    np.save(dir_hist + filename_base+'_deltaz_physical.npy', deltaz_physical)
    print(filename_base + ' saved')

    # save pair catalogs
    if save_catalog:
        cat_pair_close.write(dir_table + filename_base + '_deltaz_close.fits', overwrite=True)
        cat_pair_random.write(dir_table + filename_base + '_deltaz_random.fits', overwrite=True)
else:
    print(cat_name, 'failed', 'rank', rank, fails)



