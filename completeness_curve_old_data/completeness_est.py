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


as_func_of = sys.argv[1]  # completeness as a function of mass or magnitude
z_low = eval(sys.argv[2])
z_high = eval(sys.argv[3])
data_type = sys.argv[4]

if data_type == 'olddata':
    z_keyname = 'zKDEPeak'
else:
    z_keyname = 'Z_BEST'

cat_names = ['DEEP_deep']
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
        np.savetxt(path + 'comp_bootstrap_'+data_type+'_'+as_func_of+'.txt', all_curves)
    else:
        np.savetxt(path + 'comp_bootstrap_'+data_type+'_' + as_func_of + '_' + str(z_low)+'_'+str(z_high)+'.txt', all_curves)

# children processes (I said the calculation!)
else:
    for cat_name in cat_names:
        print('==============='+cat_name+'==================')
        mock_cat = Table.read(cat_stack_dir+'matched_'+data_type+'_cat_stack_'+cat_name+'.fits')
        print('len=='+str(len(mock_cat)))

        # keep only the mock objects (inserted as CHECK_IMAGE)
        mock_cat = mock_cat[mock_cat['ORIGINAL'] == False]
        print('Matched with phys, len=='+str(len(mock_cat)))

        # bootstrap resampling
        boot_idx = bootstrap(np.arange(len(mock_cat)), bootnum=1)
        mock_cat = mock_cat[boot_idx[0].astype(int)]

        if as_func_of == 'mag':
            bin_number = 25
            bin_edges = np.linspace(15, 30, num=bin_number)

            mag_list = np.array(mock_cat['i'])
            mag_list = mag_list[~np.isnan(mag_list)]
            all = np.histogram(mag_list, bins=bin_edges)[0]

            cat_detected = mock_cat[~np.isnan(mock_cat['FLUX_APER_1.0'])]
            mag_list_detected = np.array(cat_detected['i'])
            mag_list_detected = mag_list_detected[~np.isnan(mag_list_detected)]
            detected = np.histogram(mag_list_detected, bins=bin_edges)[0]

        elif as_func_of == 'mass':
            # redshift cut
            mock_cat = mock_cat[mock_cat[z_keyname] > z_low]
            mock_cat = mock_cat[mock_cat[z_keyname] < z_high]

            bin_number = 25
            bin_edges = np.linspace(7, 13, num=bin_number)

            mass_list = np.array(mock_cat['MASS_MED'])  # kpc
            all = np.histogram(mass_list, bins=bin_edges)[0]

            cat_detected = mock_cat[~np.isnan(mock_cat['FLUX_APER_1.0'])]
            mass_list_detected = np.array(cat_detected['MASS_MED'])  # kpc
            detected = np.histogram(mass_list_detected, bins=bin_edges)[0]

        else:
            raise ValueError('not acceptable argument for as_func_of: '+as_func_of)

        all_gals += all
        detected_gals += detected

    all_gals[all_gals==0] = 1
    all_curve = detected_gals/all_gals
    comm.send(all_curve, dest=n_procs-1)
    np.save('bin_edges.npy', bin_edges)
