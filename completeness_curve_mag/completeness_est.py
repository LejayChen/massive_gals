from astropy.table import Table
import numpy as np
from astropy.stats import bootstrap
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
n_procs = comm.Get_size()
rank = comm.Get_rank()

all_gals = 0
sf_gals = 0
q_gals = 0
detected_gals = 0
detected_sfs = 0
detected_qs = 0

sfq_type = sys.argv[1]
as_func_of = sys.argv[2]  # completeness as a function of mass or magnitude
z_low = eval(sys.argv[3]) - 0.1
z_high = eval(sys.argv[3]) + 0.1
sfq_method = sys.argv[4]

cat_names = ['COSMOS_deep', 'DEEP_deep', 'XMM-LSS_deep', 'ELAIS_deep']
cat_stack_dir = '/home/lejay/projects/def-sawicki/lejay/completeness_output_mock_cats/rand_pos/'

# parent process: collect all results

if rank == n_procs-1:
    for i in range(n_procs-1):
        all_curve = comm.recv(source=MPI.ANY_SOURCE)
        if i == 0:
            all_curves = all_curve
        else:
            all_curves = np.vstack((all_curves, all_curve))

    path = 'curves/'
    if as_func_of == 'mag':
        np.savetxt(path + 'comp_bootstrap_new_'+as_func_of+'.txt', all_curves)
    else:
        np.savetxt(path + 'comp_bootstrap_new_' + as_func_of + '_' + sfq_method + '_' + sfq_type + '_' + str(round(z_low,1)) + '_'+str(round(z_high,1))+'.txt', all_curves)

# children processes (I said the calculation!)
else:
    for cat_name in cat_names:
        print('==============='+cat_name+'==================')
        mock_cat = Table.read(cat_stack_dir+'matched_new_cat_stack_'+cat_name+'_gal_cut_params.fits')
        print('len=='+str(len(mock_cat)))

        # keep only the mock objects (inserted as CHECK_IMAGE)
        mock_cat = mock_cat[mock_cat['ORIGINAL'] == False]
        print('Matched with phys, len=='+str(len(mock_cat)))

        # selections
        if sfq_method == 'SSFR_MED':
            if sfq_type == 'sf':
                mock_cat = mock_cat[mock_cat['SSFR_MED'] > -11]
            elif sfq_type == 'q':
                mock_cat = mock_cat[mock_cat['SSFR_MED'] < -11]
        elif sfq_method == 'sfq_nuvrk' or sfq_method == 'sfq_nuvrz':
            if sfq_type == 'sf':
                mock_cat = mock_cat[mock_cat[sfq_method] == 0]
            elif sfq_type == 'q':
                mock_cat = mock_cat[mock_cat[sfq_method] == 1]
        elif sfq_method == 'sfProb_nuvrk' or sfq_method == 'sfProb_nuvrz':
            mock_cat = mock_cat[mock_cat[sfq_method] > 0]
            mock_cat = mock_cat[mock_cat[sfq_method] < 1]

        # bootstrap resampling
        boot_idx = bootstrap(np.arange(len(mock_cat)), bootnum=1)
        mock_cat = mock_cat[boot_idx[0].astype(int)]

        if as_func_of == 'mag':
            bin_number = 25
            bin_edges = np.linspace(15, 30, num=bin_number)
            mock_cat = mock_cat[~np.isnan(np.array(mock_cat['i']))]
            mag_list = np.array(mock_cat['i'])
            if 'sfProb' in sfq_method:
                if sfq_type == 'sf':
                    all = np.histogram(mag_list, bins=bin_edges, weights=mock_cat[sfq_method])[0]
                elif sfq_type == 'q':
                    all = np.histogram(mag_list, bins=bin_edges, weights=1-mock_cat[sfq_method])[0]
                else:
                    all = np.histogram(mag_list, bins=bin_edges)[0]
            else:
                all = np.histogram(mag_list, bins=bin_edges)[0]

            cat_detected = mock_cat[~np.isnan(mock_cat['FLUX_APER_1.0'])]
            mag_list_detected = np.array(cat_detected['i'])
            if 'sfProb' in sfq_method:
                if sfq_type == 'sf':
                    detected = np.histogram(mag_list_detected, bins=bin_edges, weights=cat_detected[sfq_method])[0]
                elif sfq_type == 'q':
                    detected = np.histogram(mag_list_detected, bins=bin_edges, weights=1-cat_detected[sfq_method])[0]
                else:
                    detected = np.histogram(mag_list_detected, bins=bin_edges)[0]
            else:
                detected = np.histogram(mag_list_detected, bins=bin_edges)[0]

        elif as_func_of == 'mass':
            # redshift cut
            mock_cat = mock_cat[mock_cat['ZPHOT'] > z_low]
            mock_cat = mock_cat[mock_cat['ZPHOT'] < z_high]
            mock_cat = mock_cat[~np.isnan(np.array(mock_cat['MASS_MED']))]

            bin_number = 25
            bin_edges = np.linspace(7, 13, num=bin_number)

            mass_list = np.array(mock_cat['MASS_MED'])
            if 'sfProb' in sfq_method:
                if sfq_type == 'sf':
                    all = np.histogram(mass_list, bins=bin_edges, weights=mock_cat[sfq_method])[0]
                elif sfq_type == 'q':
                    all = np.histogram(mass_list, bins=bin_edges, weights=1-mock_cat[sfq_method])[0]
                else:
                    all = np.histogram(mass_list, bins=bin_edges)[0]
            else:
                all = np.histogram(mass_list, bins=bin_edges)[0]

            cat_detected = mock_cat[~np.isnan(mock_cat['FLUX_APER_1.0'])]
            mass_list_detected = np.array(cat_detected['MASS_MED'])  # kpc
            if 'sfProb' in sfq_method:
                if sfq_type == 'sf':
                    detected = np.histogram(mass_list_detected, bins=bin_edges, weights=cat_detected[sfq_method])[0]
                elif sfq_type == 'q':
                    detected = np.histogram(mass_list_detected, bins=bin_edges, weights=1-cat_detected[sfq_method])[0]
                else:
                    detected = np.histogram(mass_list_detected, bins=bin_edges)[0]
            else:
                detected = np.histogram(mass_list_detected, bins=bin_edges)[0]

        else:
            raise ValueError('not acceptable argument for as_func_of: '+as_func_of)

        all_gals += all
        detected_gals += detected

    all_gals[all_gals == 0] = 1
    all_curve = detected_gals/all_gals
    comm.send(all_curve, dest=n_procs-1)
    np.save('bin_edges.npy', bin_edges)
