import numpy as np
import sys
from astropy.table import Table, vstack
import os
import time

path = sys.argv[1]
sat_z_cut = sys.argv[2]
cat_name = sys.argv[3]
try:
    masscut_low_host = float(sys.argv[4])
    save_cen_cat = True
except ValueError:
    masscut_low_host = sys.argv[4]
    save_cen_cat = False

masscut_low = sys.argv[5]
masscut_high = sys.argv[6]
csfq = sys.argv[7]
cat_type = sys.argv[8]

print('stacking ... ')
for z in ['0.4', '0.6', '0.8', '1.0']:
    for ssfq in ['ssf','sq','all']:
        for result_type in ['', '_sat', '_bkg']:
            file_name_base = path + 'count'+cat_name+result_type+'_'+sat_z_cut+'_'+str(masscut_low)+'_'+\
                             str(masscut_high)+'_'+str(csfq)+'_'+str(ssfq)+'_'+z
            radial_tot = np.zeros(14)
            radial_tot_err = np.zeros(14)
            n_gals_tot = 0
            for i in range(10):
                radial_with_error = np.loadtxt(file_name_base + '_' + str(i) + '.txt')
                split_index = int(((len(radial_with_error) - 1) / 2) + 1)
                n_gals = radial_with_error[0]
                radial = radial_with_error[1:split_index]
                radial_err = radial_with_error[split_index:]
                radial_tot += radial
                radial_tot_err += radial_err**2
                n_gals_tot += n_gals

            n_gals_tot = n_gals_tot
            radial_tot = radial_tot/10
            radial_tot_err = np.sqrt(radial_tot_err)/10
            radial_tot = np.append(radial_tot, radial_tot_err)
            radial_tot = np.append([n_gals_tot], radial_tot)
            np.savetxt(file_name_base+'.txt', radial_tot, header=time.asctime(time.localtime(time.time())))
            os.system('rm ' + file_name_base + '_[0-9].txt')  # remove temporary files

for z in ['0.4', '0.6', '0.8', '1.0']:
    # stack central galaxy catalog
    if csfq == 'all' and save_cen_cat and cat_type=='v9':
        can_cat_filename_base = 'central_cat/' + 'isolated_' + cat_name+'_'+str(sat_z_cut)+\
                            '_'+str(masscut_low_host) + '_' + z + '_'
        for k in range(10):
            cat_cen = Table.read(can_cat_filename_base + str(k) + '_massive.positions.fits')
            print('length of '+str(k)+' cat_cen:', len(cat_cen))
            if k == 0:
                cat_cen_stack = cat_cen
            else:
                cat_cen_stack = vstack([cat_cen_stack, cat_cen], metadata_conflicts='silent')

        cat_cen_stack.write(can_cat_filename_base + 'massive.positions.fits', overwrite=True)
        print('saved: '+can_cat_filename_base+'massive.positions.fits' )
        os.system('rm '+can_cat_filename_base + '[0-9]_massive.positions.fits')
