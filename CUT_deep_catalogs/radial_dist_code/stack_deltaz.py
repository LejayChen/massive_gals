import sys, os
import numpy as np


sample_selection=sys.argv[1]
mag_bright=sys.argv[2]
mag_faint=sys.argv[3]
pair_sfq=sys.argv[4]
z=sys.argv[5]
catalog_type=sys.argv[6]
zkeyname=sys.argv[7]
cat_name=sys.argv[8]
nProcs = 10

if sample_selection == 'mag':
    z_low = 0.0
    z_high = 7.0
elif sample_selection == 'magz':
    z_low = eval(sys.argv[5])-0.05
    z_high = eval(sys.argv[5])+0.05
elif sample_selection == 'z':
    z_low = eval(sys.argv[5])-0.05
    z_high = eval(sys.argv[5])+0.05
else:
    raise ValueError

if catalog_type == 'phos':  # phosphoros catalog
    if cat_name not in ['XMM-LSS_deep', 'COSMOS_deep']:
        zkeyname = 'ZPHOT'


filename_base_base = 'catalog_' + str(round(z_low,1)) + '_' + str(round(z_high,1)) + '_' + sample_selection + '_' + str(mag_bright) + '_' + str(mag_faint) + '_'
if pair_sfq == 'all-all':
    filename_base = filename_base_base + cat_name + '_' + catalog_type + '_' + zkeyname
else:
    filename_base = filename_base_base + pair_sfq + '_' + cat_name + '_' + catalog_type + '_' + zkeyname

pair_list_stack = 0  # stack deltaz list (not binned)
pair_hist_stack = 0  # stack binned data
dir_list = 'list_delta_z/'
dir_hist = 'delta_z/'
for pair_type in ['close', 'random', 'physical']:

    for rank in range(nProcs):
        # stack deltaz_lists
        try:
            pair_list_rank = np.load(dir_list + filename_base + '_' + str(rank) + '_deltaz_list_' + pair_type + '.npy') # unbinned raw data
            print(len(pair_list_rank),dir_list + filename_base + '_' + str(rank) + '_deltaz_list_' + pair_type + '.npy')
            if rank == 0:
                pair_list_stack = pair_list_rank.tolist()
            else:
                pair_list_stack += pair_list_rank.tolist()
        except FileNotFoundError:
            print(cat_name, mag_bright, mag_faint, rank, 'File not found.')

        # stack histograms
        try:
            pair_hist_rank = np.load(dir_hist + filename_base + '_' + str(rank) + '_deltaz_'+pair_type+'.npy') # binned data
            if rank == 0:
                pair_hist_stack = pair_hist_rank
            else:
                pair_hist_stack += pair_hist_rank
        except FileNotFoundError:
            print(cat_name, mag_bright, mag_faint, rank, 'File not found.')

    pair_hist_stack = pair_hist_stack / nProcs
    np.save(dir_list + filename_base + '_deltaz_list_' + pair_type + '.npy', pair_list_stack)
    np.save(dir_hist + filename_base + '_deltaz_' + pair_type + '.npy', pair_hist_stack)

    # remove temporary files
    os.system('rm ' + dir_list + filename_base + '_[0-9]_deltaz_list_' + pair_type + '.npy')
    os.system('rm ' + dir_hist + filename_base + '_[0-9]_deltaz_'+pair_type+'.npy')