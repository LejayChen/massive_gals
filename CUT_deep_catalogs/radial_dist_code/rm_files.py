import sys
import os
import glob
path = sys.argv[1]
masscut_low_host = sys.argv[2]
masscut_low = sys.argv[3]
masscut_high = sys.argv[4]

for cat_name in ['COSMOS_deep','DEEP_deep','ELAIS_deep','XMM-LSS_deep']:
    for z in ['0.4', '0.6', '0.8', '1.0']:
        for ssfq in ['ssf','sq','all']:
            for result_type in ['', '_sat', '_bkg']:
                file_name_base = path + 'count' + cat_name + result_type+'_*_'+str(masscut_low)+'_'+\
                             str(masscut_high)+'_*_'+str(ssfq)+'_'+z
                print(file_name_base)
                if len(glob.glob(file_name_base + '_[0-9].txt'))>0:
                    os.system('rm ' + file_name_base + '_[0-9].txt')  # remove temporary files
