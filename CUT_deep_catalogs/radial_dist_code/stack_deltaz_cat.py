import sys, os
from astropy.table import *
from mpi4py import MPI

comm=MPI.COMM_WORLD
rank = comm.Get_rank()
nProcs = comm.Get_size()

sample_selection=sys.argv[1]
mag_bright=sys.argv[2]
mag_faint=sys.argv[3]
pair_sfq=sys.argv[4]
z=sys.argv[5]
catalog_type=sys.argv[6]
zkeyname=sys.argv[7]
nProcs_photoz_run = 10

print('stacking ...')
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

cat_names = ['COSMOS_deep','XMM-LSS_deep','DEEP_deep','ELAIS_deep']
cat_name = cat_names[rank]
if catalog_type == 'phos':  # phosphoros catalog
    if cat_name not in ['XMM-LSS_deep', 'COSMOS_deep']:
        zkeyname = 'ZPHOT'

dir_cat = '/scratch/lejay/cat_delta_z/'
filename_base_base = dir_cat + 'catalog_' + str(round(z_low,1)) + '_' + str(round(z_high,1)) + '_' + sample_selection + '_' + str(mag_bright) + '_' + str(mag_faint) + '_'
if pair_sfq == 'all-all':
    filename_base = filename_base_base + cat_name + '_' + catalog_type + '_' + zkeyname
else:
    filename_base = filename_base_base + pair_sfq + '_' + cat_name + '_' + catalog_type + '_' + zkeyname


for pair_type in ['close', 'random']:
    for rank_photoz_run in range(nProcs_photoz_run):
        cat_to_stack = Table.read(filename_base+'_'+str(rank_photoz_run)+'_deltaz_'+pair_type+'.fits')
        if rank_photoz_run ==0:
            cat_stacked = cat_to_stack
        else:
            vstack([cat_stacked, cat_to_stack])

    cat_stacked.write(filename_base+'_'+pair_type+'.fits', overwrite=True)

    # remove temporary files
    os.system('rm '+filename_base+'_[0-9]_deltaz_'+pair_type+'.fits')