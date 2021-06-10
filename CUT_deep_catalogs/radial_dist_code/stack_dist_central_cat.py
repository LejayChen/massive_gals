import numpy as np
import sys
from astropy.table import Table, vstack
import os

path = sys.argv[1]
sat_z_cut = sys.argv[2]
masscut_low_host = sys.argv[3]
masscut_low = sys.argv[4]
masscut_high = sys.argv[5]
csfq = sys.argv[6]

print('stacking ... ')
for z in ['1.0']:
    for cat_name in ['DEEP_deep','COSMOS_deep','ELAIS_deep', 'XMM-LSS_deep']:
        # stack central galaxy catalog
        can_cat_filename_base = 'central_cat/' + 'isolated_' + cat_name+'_'+str(sat_z_cut)+\
                            '_'+str(masscut_low_host) + '_' + z + '_'
        for k in range(10):
            cat_cen = Table.read(can_cat_filename_base + str(k) + '_massive.positions.fits')
            print('length of '+str(k)+' cat_cen:', len(cat_cen))
            if k == 0:
                cat_cen_stack = cat_cen
            else:
                cat_cen_stack = vstack([cat_cen_stack, cat_cen], metadata_conflicts='silent')

        print(z, cat_name, len(cat_cen_stack))
        cat_cen_stack.write(can_cat_filename_base + 'massive.positions.fits', overwrite=True)
        print('saved: '+can_cat_filename_base+'massive.positions.fits' )
        os.system('rm ' + can_cat_filename_base + '[0-9]_massive.positions.fits')