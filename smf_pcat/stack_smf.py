import numpy as np
import sys
import os
import time
from astropy.table import vstack, Table

cat_names = ['COSMOS_deep','DEEP_deep','ELAIS_deep','XMM-LSS_deep']
cores_per_cat = 36 // len(cat_names)

ssfq = sys.argv[1]
z_min = sys.argv[2]
path = sys.argv[3]
scratch_path = '/scratch/lejay/'+path 
bin_number = eval(sys.argv[4])
masscut_low = eval(sys.argv[5])
masscut_host = eval(sys.argv[6])
csfq = sys.argv[7]

r_low = eval(sys.argv[8]) # Mpc
r_high = eval(sys.argv[9]) # Mpc
rel_riso = bool(eval(sys.argv[10]))
rel_mscale = bool(eval(sys.argv[11]))
r_factor = eval(sys.argv[12])

print('relative r_iso?', rel_riso)

for cat_name in cat_names:
    for result_type in ['total', 'sat', 'bkg']:
        if (not rel_riso) and (not rel_mscale):
            file_name_base = 'smf_' + cat_name + '_cen_' + str(masscut_host) +'_' + str(r_low) + '_'+str(r_high) + '_' + str(masscut_low) + '_' + csfq + '_' + ssfq + '_' + z_min + '_' + result_type
        elif rel_riso and rel_mscale:
            file_name_base = 'smf_' + cat_name + '_cen_' + str(masscut_host) + '_rel_mscale_' + str(r_low) + '_r200_' + str(masscut_low) + '_' + csfq + '_' + ssfq + '_' + z_min +  '_' + result_type
            
        elif rel_mscale:
            file_name_base = 'smf_' + cat_name + '_cen_' + str(masscut_host) +'_rel_mscale_' + str(r_low) + '_'+str(r_high) + '_' + str(masscut_low) + '_' + csfq + '_' + ssfq + '_' + z_min + '_' + result_type
        elif rel_riso:
            if r_factor != 1.0:
                r_factor_infilename = str(r_factor)
            else:
                r_factor_infilename = ''
            file_name_base = 'smf_' + cat_name + '_cen_' + str(masscut_host) + '_' + str(r_low) + '_'+r_factor_infilename+'r200_' + str(masscut_low) + '_' + csfq + '_' + ssfq + '_' + z_min + '_' + result_type

        smf_tot = np.zeros(bin_number)
        smf_tot_inf = np.zeros(bin_number)
        smf_tot_sup = np.zeros(bin_number)
        n_gals_tot = 0

        print(path + file_name_base)

        error_sq_inf = np.zeros(bin_number)
        error_sq_sup = np.zeros(bin_number)
        for i in range(cores_per_cat):
            smf_with_error = np.load(scratch_path + file_name_base + '_' + str(i) + '.npy')

            n_gals = smf_with_error[0] # number of central galaxies
            smf = smf_with_error[1:1+bin_number]
            smf_inf = smf_with_error[1+bin_number:1+2*bin_number]
            smf_sup = smf_with_error[1+2*bin_number:1+3*bin_number]

            # SMFs are stacked, not divided by n_gal at this point
            smf_tot += smf   # stacked smf
            error_sq_inf += (smf-smf_inf)**2
            error_sq_sup += (smf_sup-smf)**2
            n_gals_tot += n_gals
            # print((smf-smf_inf)/smf)

        smf_tot = smf_tot / n_gals_tot # smf per central
        smf_tot_inf = smf_tot - np.sqrt(error_sq_inf) / n_gals_tot
        smf_tot_sup = smf_tot + np.sqrt(error_sq_sup) / n_gals_tot

        print('')
        print('######')
        print(cat_name,result_type, round(sum(smf_tot),1), n_gals_tot)
        print(smf_tot)
        print(smf_tot_inf)

        smf_tot = np.append(smf_tot, smf_tot_inf)
        smf_tot = np.append(smf_tot, smf_tot_sup)
        smf_tot = np.append([n_gals_tot], smf_tot)
        np.savetxt(path + file_name_base+'.txt', smf_tot, header=time.asctime(time.localtime(time.time())))

        print(cat_name, result_type, 'stacked')
        # os.system('rm ' + scratch_path + file_name_base + '_[0-9].npy')  # remove temporary files

    # stack central catalog
    if csfq == 'all' and ssfq == 'all':
        file_name_base_cen =  '_cen_' + str(masscut_host) + '_' + z_min 
        for i in range(cores_per_cat):
            cen_cat = Table.read(scratch_path + cat_name + file_name_base_cen + '_' + str(i)  + '.fits')
            if i==0:
                cen_cat_combine = cen_cat
            else:
                cen_cat_combine = vstack([cen_cat_combine,cen_cat])
        cen_cat_combine.write(path + cat_name + file_name_base_cen + '.fits', overwrite=True)
        # os.system('rm ' + scratch_path + cat_name + '_'+ file_name_base_cen + '_[0-9].fits')  # remove temporary files

