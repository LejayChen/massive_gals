import numpy as np
import sys
import os
import time

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

for cat_name in cat_names:
    for result_type in ['total', 'sat', 'bkg']:
        file_name_base = 'smf_' + cat_name + '_cen_' + str(masscut_host) +'_' + str(r_low) + '_'+str(r_high) + '_' + str(masscut_low) + '_' + csfq + '_' + ssfq + '_' + z_min + '_' + result_type
        smf_tot = np.zeros(bin_number)
        smf_tot_inf = np.zeros(bin_number)
        smf_tot_sup = np.zeros(bin_number)
        n_gals_tot = 0

        print(path + file_name_base)

        error_inf = np.zeros(bin_number)
        error_sup = np.zeros(bin_number)
        for i in range(cores_per_cat):
            smf_with_error = np.load(scratch_path + file_name_base + '_' + str(i) + '.npy')

            n_gals = smf_with_error[0] # number of central galaxies
            smf = smf_with_error[1:1+bin_number]
            smf_inf = smf_with_error[1+bin_number:1+2*bin_number]
            smf_sup = smf_with_error[1+2*bin_number:1+3*bin_number]
            # print('num of gals in rank', i, cat_name, result_type, round(sum(smf)/n_gals,1),n_gals,len(smf))

            # SMFs are stacked, not divided by n_gal at this point
            smf_tot += smf   # stacked smf
            error_inf += (smf-smf_inf)**2
            error_sup += (smf_sup-smf)**2
            n_gals_tot += n_gals

        smf_tot = smf_tot / n_gals_tot # smf per central
        smf_tot_inf = smf_tot - np.sqrt(error_inf) / n_gals_tot
        smf_tot_sup = smf_tot + np.sqrt(error_sup) / n_gals_tot
        
        print('')
        print(cat_name,result_type, round(sum(smf_tot),1), n_gals_tot)

        smf_tot = np.append(smf_tot, smf_tot_inf)
        smf_tot = np.append(smf_tot, smf_tot_sup)
        smf_tot = np.append([n_gals_tot], smf_tot)
        np.savetxt(path + file_name_base+'.txt', smf_tot, header=time.asctime(time.localtime(time.time())))

        print(cat_name, result_type, 'stacked')
        # os.system('rm ' + scratch_path + file_name_base + '_[0-9].npy')  # remove temporary files

    # stack central mass list
    # if ssfq == 'all':
    #     cen_mass_list_stack = np.array([])
    #     file_name_base_cen =  str(r_low) + '_'+str(r_high) + '_' + str(masscut_low) + '_' + str(csfq) + '_' + z_min 
    #     for i in range(cores_per_cat):
    #         cen_mass_list = np.load(scratch_path + cat_name + '_'+ file_name_base_cen + '_cen_mass_' + str(i) + '.npy')
    #         cen_mass_list_stack = np.append(cen_mass_list_stack,cen_mass_list)
    #     np.save(path + cat_name + '_cen_masslist_' + str(masscut_host) + '_'+ file_name_base_cen + '.npy', cen_mass_list_stack)
        # os.system('rm ' + scratch_path + cat_name + '_'+ file_name_base_cen + '_cen_mass_[0-9].npy')  # remove temporary files

