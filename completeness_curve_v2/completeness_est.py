from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from astropy.stats import bootstrap
from astropy.cosmology import WMAP9
import numpy as np
# import matplotlib.pyplot as plt
import sys
from mpi4py import MPI
comm = MPI.COMM_WORLD
n_procs = comm.Get_size()
rank = comm.Get_rank()

all_gals = 0
sf_gals = 0
q_gals = 0
detected_gals = 0
detected_sfs = 0
detected_qs = 0

masscut_low = float(sys.argv[1])
masscut_high = float(sys.argv[2])
z_low = round(float(sys.argv[3])-0.1, 1)
z_high = round(float(sys.argv[3])+0.1, 1)
cat_names = ['COSMOS_deep']
csfq = 'all'

# parent process: collect all results
if rank == n_procs-1:
    for i in range(n_procs-1):
        all_curve = comm.recv(source=MPI.ANY_SOURCE)
        if i == 0:
            all_curves = all_curve
        else:
            all_curves = np.vstack((all_curves, all_curve))

    path = 'curves/'
    np.savetxt(path + 'comp_bootstrap_'+csfq+'_all_'+str(masscut_low)+'_'+str(masscut_high)+'_'+str(z_low)+'_'+str(z_high)+'.txt', all_curves)

# children processes (I said the calculation!)
else:
    for cat_name in cat_names:
        print('==============='+cat_name+'==================')
        for z in [0.4, 0.6, 0.8]:
            cat_massive_gal = Table.read('/home/lejay/v2_matched_cats/central_'+cat_name+'_'+str(z)+'.fits')
            mock_cat = Table.read('./matched_cat/matched_20random_cat_stack_'+cat_name+'_'+str(z)+'.fits')
            mock_cat = mock_cat[mock_cat['Z_BEST'] > z_low]
            mock_cat = mock_cat[mock_cat['Z_BEST'] < z_high]
            print('len=='+str(len(mock_cat)))

            mock_cat = mock_cat[mock_cat['ORIGINAL'] == False]  # shifted random mock objs
            print('Matched with phys, len=='+str(len(mock_cat)))

            # mag cut / mass cut
            if z == 0.4:
                mock_cat = mock_cat[mock_cat['ri'] > 24]
            elif z == 0.6:
                mock_cat = mock_cat[mock_cat['i'] > 24.91]
            elif z == 0.8:
                mock_cat = mock_cat[mock_cat['z'] > 25.55]
            print('After mag cut, len=='+str(len(mock_cat)))

            #### MISSING MASS CUT RIGHT NOW

            bin_number = 14
            bin_edges = 10 ** np.linspace(1.0, np.log10(700), num=bin_number + 1)
            gal_ids = open('Gal_ids/gal_ids_'+cat_name+'_'+str(z)+'.txt').readlines()
            print('No. of centrals:', len(gal_ids))

            for i in range(len(gal_ids)):
                gal_ids[i] = gal_ids[i].rstrip()
                # gal_ids[i] = gal_ids[i].replace('cutout_', '')
                # gal_ids[i] = gal_ids[i].replace('.fits', '')

            for gal in cat_massive_gal:
                if gal['ID'] in gal_ids:
                    dis = WMAP9.angular_diameter_distance(float(gal['z'])).value

                    # selection for potential satellites
                    cat_neighbors = mock_cat[abs(mock_cat['X_WORLD'] - gal['RA']) < 2.5 / dis / np.pi * 180]
                    cat_neighbors = cat_neighbors[abs(cat_neighbors['Y_WORLD'] - gal['DEC']) < 2.5 / dis / np.pi * 180]
                    if len(cat_neighbors) > 0:
                        boot_idx = bootstrap(np.arange(len(cat_neighbors)), bootnum=1)
                        cat_neighbors = cat_neighbors[boot_idx[0].astype(int)]
                    else:
                        print(gal['ID'], 'len(cat_neighbors)', len(cat_neighbors))
                        continue

                    coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)
                    coord_neighbors = SkyCoord(cat_neighbors['X_WORLD'] * u.deg, cat_neighbors['Y_WORLD'] * u.deg)
                    radius_list = coord_neighbors.separation(coord_gal).degree / 180. * np.pi * dis * 1000  # kpc
                    all = np.histogram(radius_list, bins=bin_edges)[0]

                    cat_detected = cat_neighbors[~np.isnan(cat_neighbors['FLUX_APER_1.0'])]
                    coord_detected = SkyCoord(cat_detected['X_WORLD'] * u.deg, cat_detected['Y_WORLD'] * u.deg)
                    radius_list_detected = coord_detected.separation(coord_gal).degree / 180. * np.pi * dis * 1000  # kpc
                    detected = np.histogram(radius_list_detected, bins=bin_edges)[0]

                    all_gals += all
                    detected_gals += detected


    all_curve = detected_gals/all_gals
    comm.send(all_curve, dest=n_procs-1)
    np.save('bin_edges.npy', bin_edges)
