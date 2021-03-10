from astropy.table import Table
import numpy as np
from astropy.stats import bootstrap
from mpi4py import MPI
import astropy.units as u
import sys
from astropy.cosmology import WMAP9
from astropy.coordinates import SkyCoord


comm = MPI.COMM_WORLD
n_procs = comm.Get_size()
rank = comm.Get_rank()

as_func_of = 'radius'
masscut_low = float(sys.argv[1])
masscut_high = float(sys.argv[2])
z = eval(sys.argv[3])
z_low = round(float(sys.argv[3])-0.1, 1)
z_high = round(float(sys.argv[3])+0.1, 1)
sfq_method = sys.argv[4]
sfq_type = sys.argv[5]

if as_func_of == 'mass':
    bin_number = 20
    bin_edges = np.linspace(7, 12, num=bin_number + 1)
else:
    bin_number = 14
    bin_edges = 10 ** np.linspace(1.0, np.log10(700), num=bin_number + 1)

all_gals = np.zeros(bin_number)
sf_gals = np.zeros(bin_number)
q_gals = np.zeros(bin_number)
detected_gals = np.zeros(bin_number)
detected_sfs = np.zeros(bin_number)
detected_qs = np.zeros(bin_number)

cat_names = ['COSMOS_deep', 'DEEP_deep', 'XMM-LSS_deep', 'ELAIS_deep']
csfq = 'all'
cat_stack_dir = 'Output_cats/'

# parent process: collect all results
if rank == n_procs-1:
    for i in range(n_procs-1):
        all_curve = comm.recv(source=MPI.ANY_SOURCE)
        if i == 0:
            all_curves = all_curve
        else:
            all_curves = np.vstack((all_curves, all_curve))

    path = 'curves/'
    np.savetxt(path + 'comp_bootstrap_' + + as_func_of + '_'+ csfq + '_'+sfq_type+'_' + sfq_method + '_' +
                   str(masscut_low) + '_' + str(masscut_high) + '_' + str(z_low) + '_' + str(z_high) + '.txt', all_curves)

# children processes (I said the calculation!)
else:
    for cat_name in cat_names:
        print('==============='+cat_name+'==================')
        cat_massive_gal = Table.read('/home/lejay/radial_dist_code/central_cat/isolated_' + cat_name + '_11.15_' + str(z) + '_massive.positions.fits')
        mock_cat = Table.read(cat_stack_dir+'phys_matched_new_cat_stack_'+cat_name+'_'+str(z)+'.fits')
        print('len=='+str(len(mock_cat)))

        # cleaning
        mock_cat = mock_cat[mock_cat['ORIGINAL'] == False]  # keep only the mock objects (inserted as CHECK_IMAGE)
        print('Matched with phys, len==' + str(len(mock_cat)))
        mock_cat = mock_cat[~np.isnan(np.array(mock_cat['MASS_MED']))]  # clean the catalog for mass measurement
        mock_cat = mock_cat[~np.isnan(np.array(mock_cat['ZPHOT']))]
        mock_cat = mock_cat[~np.isnan(np.array(mock_cat['RA']))]
        mock_cat = mock_cat[~np.isnan(np.array(mock_cat['X_WORLD']))]

        # redshift cut for massive galaxies
        cat_massive_gal = cat_massive_gal[~np.isnan(cat_massive_gal['ZPHOT'])]
        cat_massive_gal = cat_massive_gal[~np.isnan(cat_massive_gal['MASS_MED'])]
        cat_massive_gal = cat_massive_gal[cat_massive_gal['ZPHOT'] > z_low]
        cat_massive_gal = cat_massive_gal[cat_massive_gal['ZPHOT'] < z_high]

        gal_ids = open('Gal_ids/gal_ids_' + cat_name + '_' + str(z) + '.txt').readlines()
        for i in range(len(gal_ids)):
            gal_ids[i] = gal_ids[i].rstrip()
        print('No. of centrals:', len(gal_ids), len(cat_massive_gal))

        # sfq selections
        if sfq_method == 'SSFR_MED':
            if sfq_type == 'ssf':
                mock_cat = mock_cat[mock_cat['SSFR_MED'] > -11]
            elif sfq_type == 'sq':
                mock_cat = mock_cat[mock_cat['SSFR_MED'] < -11]
            else:
                mock_cat = mock_cat[~np.isnan(mock_cat['SSFR_MED'])]

        elif sfq_method == 'sfq_nuvrk' or sfq_method == 'sfq_nuvrz':
            if sfq_type == 'ssf':
                mock_cat = mock_cat[mock_cat[sfq_method] == 0]
            elif sfq_type == 'sq':
                mock_cat = mock_cat[mock_cat[sfq_method] == 1]
            else:
                mock_cat = mock_cat[np.logical_or(mock_cat[sfq_method] == 1, mock_cat[sfq_method] == 0)]

        elif sfq_method == 'sfProb_nuvrk' or sfq_method == 'sfProb_nuvrz':
            mock_cat = mock_cat[mock_cat[sfq_method] > 0]
            mock_cat = mock_cat[mock_cat[sfq_method] < 1]

        # bootstrap re-sampling
        # boot_idx = bootstrap(np.arange(len(mock_cat)), bootnum=1)
        # mock_cat = mock_cat[boot_idx[0].astype(int)]
        print('number of mock gals:', len(mock_cat))

        gal_count = 0
        for gal in cat_massive_gal:
            if str(gal['ID']) in gal_ids:
                dis = WMAP9.angular_diameter_distance(float(gal['ZPHOT'])).value
                coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)

                # selection for potential satellites (in mock catalog)
                cat_neighbors = mock_cat[abs(mock_cat['X_WORLD'] - gal['RA']) < 2.5 / dis / np.pi * 180]
                cat_neighbors = cat_neighbors[abs(cat_neighbors['Y_WORLD'] - gal['DEC']) < 2.5 / dis / np.pi * 180]
                cat_neighbors = cat_neighbors[abs(cat_neighbors['ZPHOT'] - gal['ZPHOT']) < 4.5 * 0.044 * (1 + gal['ZPHOT'])]

                coord_neighbors = SkyCoord(cat_neighbors['X_WORLD'] * u.deg, cat_neighbors['Y_WORLD'] * u.deg)
                radius_list = coord_neighbors.separation(coord_gal).degree / 180. * np.pi * dis * 1000  # kpc
                cat_neighbors = cat_neighbors[radius_list < 700]

                if as_func_of == 'mass':
                    mass_list = cat_neighbors['MASS_MED']
                    if 'sfProb' in sfq_method:
                        if sfq_type == 'ssf':
                            all_gal = np.histogram(mass_list, bins=bin_edges, weights=cat_neighbors[sfq_method])[0]
                        elif sfq_type == 'sq':
                            all_gal = np.histogram(mass_list, bins=bin_edges, weights=1 - cat_neighbors[sfq_method])[0]
                        else:
                            all_gal = np.histogram(mass_list, bins=bin_edges)[0]
                    else:
                        all_gal = np.histogram(mass_list, bins=bin_edges)[0]

                    # re-detected galaxies
                    cat_neighbors_detected = cat_neighbors[~np.isnan(cat_neighbors['FLUX_APER_1.0'])]
                    mass_list_detected = cat_neighbors_detected['MASS_MED']
                    if 'sfProb' in sfq_method:
                        if sfq_type == 'ssf':
                            detected = np.histogram(mass_list_detected, bins=bin_edges, weights=cat_neighbors_detected[sfq_method])[0]
                        elif sfq_type == 'sq':
                            detected = np.histogram(mass_list_detected, bins=bin_edges, weights=1-cat_neighbors_detected[sfq_method])[0]
                        else:
                            detected = np.histogram(mass_list_detected, bins=bin_edges)[0]
                    else:
                        detected = np.histogram(mass_list_detected, bins=bin_edges)[0]

                else:
                    # mass cut
                    cat_neighbors = cat_neighbors[cat_neighbors['MASS_MED'] > masscut_low]
                    cat_neighbors = cat_neighbors[cat_neighbors['MASS_MED'] < masscut_high]

                    # all mock galaxies
                    if 'sfProb' in sfq_method:
                        if sfq_type == 'ssf':
                            all_gal = np.histogram(radius_list, bins=bin_edges, weights=cat_neighbors[sfq_method])[0]
                        elif sfq_type == 'sq':
                            all_gal = np.histogram(radius_list, bins=bin_edges, weights=1-cat_neighbors[sfq_method])[0]
                        else:
                            all_gal = np.histogram(radius_list, bins=bin_edges)[0]
                    else:
                        all_gal = np.histogram(radius_list, bins=bin_edges)[0]

                    # re-detected galaxies
                    cat_neighbors_detected = cat_neighbors[~np.isnan(cat_neighbors['FLUX_APER_1.0'])]
                    coord_neighbors_detected = SkyCoord(cat_neighbors_detected['X_WORLD'] * u.deg, cat_neighbors_detected['Y_WORLD'] * u.deg)
                    radius_list_detected = coord_neighbors_detected.separation(coord_gal).degree / 180. * np.pi * dis * 1000 # kpc
                    if 'sfProb' in sfq_method:
                        if sfq_type == 'ssf':
                            detected = np.histogram(radius_list_detected, bins=bin_edges, weights=cat_neighbors_detected[sfq_method])[0]
                        elif sfq_type == 'sq':
                            detected = np.histogram(radius_list_detected, bins=bin_edges, weights=1-cat_neighbors_detected[sfq_method])[0]
                        else:
                            detected = np.histogram(radius_list_detected, bins=bin_edges)[0]
                    else:
                        detected = np.histogram(radius_list_detected, bins=bin_edges)[0]

                all_gals += all_gal
                detected_gals += detected
                gal_count += 1

                # checkpoint
                print('======'+cat_name+'=='+sfq_method+'=='+sfq_type+'=='+str(round(z_low+0.1, 1))+'=='+str(gal['ID'])+'=====')
                print(sum(detected), detected)
                print(sum(all_gal), all_gal)

        print('total central gal counted:', gal_count, cat_name)

    all_gals[all_gals == 0] = 1
    print('======' + cat_name + '==' + sfq_method + '==' + sfq_type + '==' + str(round(z_low + 0.1, 1)) + '======')
    print('detected', sum(detected_gals), detected_gals)
    print('total   ', sum(detected_gals), all_gals)

    all_curve = detected_gals/all_gals
    print('total curve', all_curve)
    np.save('bin_edges.npy', bin_edges)
    comm.send(all_curve, dest=n_procs-1)
