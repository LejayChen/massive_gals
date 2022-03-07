import numpy as np
import sys
from astropy.table import Table, vstack
import os
import time

path = sys.argv[1]
sat_z_cut = sys.argv[2]
cat_name = sys.argv[3]
save_catalogs = True

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
bin_number = int(sys.argv[9])

if 'inside_j' in path:
    inside_j = True
else:
    inside_j = False

print('stacking ... ')
for z in ['0.4', '0.6', '0.8', '1.0']:  # stack radial number density results
    for ssfq in ['ssf','sq','all']:
        for result_type in ['', '_sat', '_bkg']:
            file_name_base = path + 'count'+cat_name+result_type+'_'+sat_z_cut+'_'+str(masscut_low)+'_'+\
                             str(masscut_high)+'_'+str(csfq)+'_'+str(ssfq)+'_'+z
            radial_tot = np.zeros(bin_number)
            radial_tot_err = np.zeros(bin_number)
            n_gals_tot = 0
            for i in range(10):
                try:
                    radial_with_error = np.loadtxt(file_name_base + '_' + str(i) + '.txt')
                except FileNotFoundError:
                    continue
                split_index = int(((len(radial_with_error) - 1) / 2) + 1)
                n_gals = radial_with_error[0]  # number of central galaxies
                radial = radial_with_error[1:split_index]
                radial_err = radial_with_error[split_index:]

                radial_tot += radial*n_gals
                radial_tot_err += radial_err**2*n_gals**2
                n_gals_tot += n_gals

            radial_tot = radial_tot/n_gals_tot
            radial_tot_err = np.sqrt(radial_tot_err)/n_gals_tot
            radial_tot = np.append(radial_tot, radial_tot_err)
            radial_tot = np.append([n_gals_tot], radial_tot)
            np.savetxt(file_name_base+'.txt', radial_tot, header=time.asctime(time.localtime(time.time())))
            print('saved: ' + file_name_base+'.txt')
            os.system('rm ' + file_name_base + '_[0-9].txt')  # remove temporary files

for z in ['0.4', '0.6', '0.8', '1.0']:  # stack central galaxy catalog
    if csfq == 'all' and save_cen_cat and cat_type == 'v9':
        can_cat_filename_base = 'central_cat/' + 'isolated_' + cat_name+'_'+str(sat_z_cut)+'_'+str(masscut_low_host) + '_' + z + '_'
        cat_cen_stack = []
        for k in range(10):
            try:
                if inside_j:
                    affix = 'massive.positions_inside_j'
                else:
                    affix = 'massive.positions'
                cat_cen = Table.read(can_cat_filename_base + str(k) + '_'+affix+'_.fits')
            except FileNotFoundError:
                continue

            if len(cat_cen_stack)==0:
                cat_cen_stack = cat_cen
            else:
                cat_cen_stack = vstack([cat_cen_stack, cat_cen], metadata_conflicts='silent')

        cat_cen_stack.write(can_cat_filename_base + 'massive.positions.fits', overwrite=True)
        print('saved: '+can_cat_filename_base+'massive.positions.fits')
        os.system('rm '+can_cat_filename_base + '[0-9]_massive.positions.fits')

for z in ['0.4', '0.6', '0.8', '1.0']:  # stack satellite /bkg objects catalog
    if csfq == 'all' and save_catalogs == True:
        sat_cat_dir = path + cat_name + '_' + str(int(eval(z) * 10))
        cat_sat_stack = []
        cat_nkg_stack = []
        for j in range(10):
            try:
                cat_sat = Table.read(sat_cat_dir+'/'+'satellites_'+cat_name+'_'+z+'_'+str(j)+'.fits')
                cat_bkg = Table.read(sat_cat_dir+'/'+'background_'+cat_name+'_'+z+'_'+str(j)+'.fits')
            except FileNotFoundError:
                continue
            if len(cat_sat_stack)==0 or len(cat_bkg_stack)==0:
                cat_sat_stack = cat_sat
                cat_bkg_stack = cat_bkg
            else:
                cat_sat_stack = vstack([cat_sat_stack, cat_sat], metadata_conflicts='silent')
                cat_bkg_stack = vstack([cat_bkg_stack, cat_bkg], metadata_conflicts='silent')

        cat_sat_stack.write(path+'satellites_'+cat_name+'_'+z+'.fits', overwrite=True)
        cat_bkg_stack.write(path+'background_'+cat_name+'_'+z+'.fits', overwrite=True)
        print('saved: satellites[/background]'+cat_name+'_'+z+'.fits')
        os.system('rm ' + sat_cat_dir + '/' + 'satellites_' + cat_name + '_' + z + '_[0-9].fits')
        os.system('rm ' + sat_cat_dir + '/'+'background_'+cat_name + '_'+z+'_[0-9].fits')
