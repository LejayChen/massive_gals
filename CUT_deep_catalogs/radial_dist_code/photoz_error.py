import sys
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import *
from mpi4py import MPI
import time

comm=MPI.COMM_WORLD
rank = comm.Get_rank()
nProcs = comm.Get_size()

cat_name = sys.argv[1]
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

cat = cat[cat['MASK'] == 0]  # unmasked
cat_gal = cat[cat['OBJ_TYPE'] == 0]  # galaxies
cat_gal = cat_gal[cat_gal['MASS_MED'] > 9.5]  # Mass selection

sfq = sys.argv[2]
if sfq == 'sf':
    cat_gal = cat_gal[cat_gal['sfProb_nuvrk'] > 0.5]  # Mass selection
elif sfq == 'q':
    cat_gal = cat_gal[cat_gal['sfProb_nuvrk'] < 0.5]  # Mass selection

for z in [0.4, 0.6, 0.8, 1.0]:
    print('')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),z)
    # load the central gal catalog
    cat_central = Table.read('central_cat/' + 'isolated_' + cat_name + '_3.0_11.3_' + str(z) +'_massive.positions.fits')

    f_closepair = open('delta_z/cen_sat_pair/delta_z_close_pair_' + cat_name + '_' + str(z) + '_' + sfq+'.txt', 'w')
    f_randompair = open('delta_z/cen_sat_pair/delta_z_random_pair_' + cat_name + '_' + str(z) + '_' + sfq+'.txt', 'w')
    for gal in cat_central:
        coord_gal = SkyCoord(gal[ra_key] * u.deg, gal[dec_key] * u.deg)

        # delta z of the central-close pairs
        cat_neighbors = cat_gal[abs(cat_gal[ra_key] - gal[ra_key]) < 25/3600.]
        cat_neighbors = cat_neighbors[abs(cat_neighbors[dec_key] - gal[dec_key]) < 25/3600.]  # circular aperture cut

        coord_neighbors = SkyCoord(np.array(cat_neighbors[ra_key]) * u.deg, np.array(cat_neighbors[dec_key]) * u.deg)
        cat_neighbors = cat_neighbors[coord_neighbors.separation(coord_gal).degree < 15/3600.]
        coord_neighbors = SkyCoord(np.array(cat_neighbors[ra_key]) * u.deg, np.array(cat_neighbors[dec_key]) * u.deg)
        cat_neighbors = cat_neighbors[coord_neighbors.separation(coord_gal).degree > 2.5/3600.]

        cat_neighbors = cat_neighbors[cat_neighbors[id_keyname] != gal[id_keyname]]
        delta_z_list = cat_neighbors[z_keyname]-gal[z_keyname]

        # delta z of the central-random pairs
        cat_rand = np.random.choice(cat_gal, len(delta_z_list), replace=False)
        delta_z_list_rand = cat_rand[z_keyname] - gal[z_keyname]

        # output: list of delta_z
        f_closepair.write(np.array_str(delta_z_list).strip('[]')+'\n')
        f_randompair.write(np.array_str(delta_z_list_rand).strip('[]')+'\n')

    f_closepair.close()
    f_randompair.close()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),z, cat_name, 'done')