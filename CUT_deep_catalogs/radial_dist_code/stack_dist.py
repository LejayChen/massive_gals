import numpy as np
import sys
from astropy.table import Table, vstack
import os

path = sys.argv[1]
sat_z_cut = sys.argv[2]
cat_name = sys.argv[3]
masscut_low_host = sys.argv[4]
masscut_low = sys.argv[5]
csfq = 'all'
ssfq = 'all'

for z in ['0.4', '0.6', '0.8']:
    file_name_base = path + 'count' + cat_name + '_' + sat_z_cut + '_' + str(masscut_low) + '_' + \
                         str(csfq) + '_' + str(ssfq) + '_' + z
    radial_tot = np.zeros(14)
    radial_tot_err = np.zeros(14)
    n_gals_tot = 0

    for i in range(10):
        cat_cen = Table.read('central_cat/'+'isolated_'+cat_name+'_'+str(masscut_low_host)+'_'+
                                 z+'_'+str(i)+'_massive.positions.fits')
        file_name_i = file_name_base + '_' + str(i) + '.txt'
        radial_with_error = np.loadtxt(file_name_i)
        split_index = int(((len(radial_with_error) - 1) / 2) + 1)

        n_gals = radial_with_error[0]
        radial = radial_with_error[1:split_index]
        radial_err = radial_with_error[split_index:]

        radial_tot += radial
        radial_tot_err += radial_err**2
        n_gals_tot += n_gals

        if i==0:
            cat_cen_stack = cat_cen
        else:
            cat_cen_stack = vstack([cat_cen_stack, cat_cen], metadata_conflicts='silent')

    n_gals_tot = n_gals_tot/10
    radial_tot = radial_tot/10
    radial_tot_err = np.sqrt(radial_tot_err)/10
    radial_tot = np.append(radial_tot, radial_tot_err)
    radial_tot = np.append([n_gals_tot], radial_tot)

    np.savetxt(file_name_base+'.txt', radial_tot)
    cat_cen_stack.write('central_cat/'+'isolated_'+cat_name+'_'+str(masscut_low_host)+'_'+
                                 z+'_'+'_massive.positions.fits', overwrite=True)
    print(file_name_base+'.txt saved.')

    # remove temporary files
    os.system('rm '+ file_name_base + '_[0-9].txt')
    os.system('rm ' + 'central_cat/'+'isolated_'+cat_name+'_'+str(masscut_low_host)+'_'+z+'_[0-9]_massive.positions.fits')
