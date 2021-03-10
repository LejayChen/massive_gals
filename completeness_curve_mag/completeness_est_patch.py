from astropy.table import Table
import numpy as np
from scipy.optimize import curve_fit
from astropy.stats import bootstrap
from mpi4py import MPI
import sys
comm = MPI.COMM_WORLD
n_procs = comm.Get_size()
rank = comm.Get_rank()


def sigmoid(x,a,b):
    return 1/(1+np.exp(-x/a+b))


sfq_type = sys.argv[1]
as_func_of = sys.argv[2]  # completeness as a function of mass or magnitude

cat_names = ['COSMOS_deep', 'DEEP_deep', 'XMM-LSS_deep', 'ELAIS_deep']
cat_stack_dir = '/home/lejay/projects/def-sawicki/lejay/completeness_output_mock_cats/rand_pos/'
for cat_name in cat_names:
    print('==============='+cat_name+'==================')
    mock_cat = Table.read(cat_stack_dir+'matched_new_cat_stack_'+cat_name+'_gal_cut_params.fits')
    print('len=='+str(len(mock_cat)))

    # keep only the mock objects (inserted as CHECK_IMAGE)
    mock_cat = mock_cat[mock_cat['ORIGINAL'] == False]
    print('Matched with phys, len=='+str(len(mock_cat)))

    # make a catalog
    cat = Table(names=['TRACT', 'PATCH', 'RA','DEC', '80p_z4_mass', '80p_z6_mass', '80p_z8_mass'], dtype=['i8','a3','f8','f8','f8','f8','f8'])
    cat_patches = Table.read('/home/lejay/tract_patches/'+cat_name.replace('_deep', '_dud')+'_patches.fits')
    tracts = np.unique(cat_patches['tract'])

    # allocate tracts to cores
    nEach = len(tracts) // n_procs
    if rank == n_procs - 1:
        my_tracts = tracts[rank * nEach:]
    else:
        my_tracts = tracts[rank * nEach:rank * nEach + nEach]

    # selection on patch
    print('rank', rank, np.array(my_tracts))
    for tract in my_tracts:
        patches = cat_patches[cat_patches['tract']==tract]['patch']
        for patch in patches:
            mock_cat_tract = mock_cat[mock_cat['TRACT_insert']==tract]
            mock_cat_patch = mock_cat_tract[mock_cat_tract['PATCH_insert']==patch]

            tract_entries = cat_patches[cat_patches['tract']==tract]
            patch_entry = tract_entries[tract_entries['patch']==patch]
            patch_cen_ra = np.array(patch_entry['center'])[0][0]
            patch_cen_dec = np.array(patch_entry['center'])[0][1]

            if len(mock_cat_patch) == 0:
                print('zero length catalog in ', tract, patch)
                continue
            else:
                # selections
                if sfq_type == 'sf':
                    mock_cat_patch = mock_cat_patch[mock_cat_patch['SSFR_MED'] > -11]
                elif sfq_type == 'q':
                    mock_cat_patch = mock_cat_patch[mock_cat_patch['SSFR_MED'] < -11]
                else:
                    pass

                # # bootstrap resampling
                # boot_idx = bootstrap(np.arange(len(mock_cat)), bootnum=1)
                # mock_cat = mock_cat[boot_idx[0].astype(int)]

                if as_func_of == 'mag':
                    bin_number = 25
                    bin_edges = np.linspace(15, 30, num=bin_number)

                    mag_list = np.array(mock_cat_patch['i'])
                    mag_list = mag_list[~np.isnan(mag_list)]
                    all = np.histogram(mag_list, bins=bin_edges)[0]

                    cat_detected = mock_cat_patch[~np.isnan(mock_cat_patch['FLUX_APER_1.0'])]
                    mag_list_detected = np.array(cat_detected['i'])
                    mag_list_detected = mag_list_detected[~np.isnan(mag_list_detected)]
                    detected = np.histogram(mag_list_detected, bins=bin_edges)[0]

                elif as_func_of == 'mass':
                    mass_80 = []
                    for z in [0.4, 0.6, 0.8]:
                        z_low = z-0.1
                        z_high = z+0.1
                        bin_number = 15
                        bin_edges = np.linspace(7.5, 10.5, num=bin_number)

                        # redshift cut
                        mock_cat_patch = mock_cat_patch[mock_cat_patch['ZPHOT'] > z_low]
                        mock_cat_patch = mock_cat_patch[mock_cat_patch['ZPHOT'] < z_high]
                        if len(mock_cat_patch) == 0:
                            print('zero length catalog in ', tract, patch)
                            mass_80.append(7.0)
                            continue

                        mass_list = np.array(mock_cat_patch['MASS_MED'])  # kpc
                        mass_list = mass_list[~np.isnan(mass_list)]
                        all = np.histogram(mass_list, bins=bin_edges)[0]

                        cat_detected = mock_cat_patch[~np.isnan(mock_cat_patch['FLUX_APER_1.0'])]
                        mass_list_detected = np.array(cat_detected['MASS_MED'])  # kpc
                        mass_list_detected = mass_list_detected[~np.isnan(mass_list_detected)]
                        detected = np.histogram(mass_list_detected, bins=bin_edges)[0]

                        all[all==0]=1
                        curve = detected / all

                        np.set_printoptions(precision=2)
                        print(tract, patch, 'detected', detected)
                        print(tract, patch, 'all', all)
                        print(tract, patch, curve)
                        for j in range(len(curve)):
                            if j < len(curve) / 2 and np.isnan(curve[j]):
                                curve[j] = 0
                            elif j > len(curve) / 2 and np.isnan(curve[j]):
                                curve[j] = 1

                        # fit with sigmoid function
                        bin_centers = np.diff(bin_edges) + bin_edges[:-1]
                        try:
                            popt, pcov = curve_fit(sigmoid, bin_centers, curve)
                        except RuntimeError:
                            mass_80.append(7.0)
                            continue

                        # find the 80% completeness threshold
                        mass_linspace = np.linspace(7, 10.5, 100)
                        comp_linspace = sigmoid(mass_linspace, popt[0], popt[1])
                        mass_80_z = mass_linspace[np.argmin(abs(comp_linspace-0.8))]

                        # append the completeness in the patch
                        mass_80.append(mass_80_z)

                    # write in the catalog
                    print(patch_cen_ra, patch_cen_dec, mass_80)
                    comp_entry = [tract, patch, patch_cen_ra, patch_cen_dec, mass_80[0], mass_80[1], mass_80[2]]
                    cat.add_row(comp_entry)

                else:
                    raise ValueError('not acceptable argument for as_func_of: '+as_func_of)

    cat.write('Output_cats/comp_patch_'+cat_name+'_'+str(rank)+'.fits', overwrite=True)



