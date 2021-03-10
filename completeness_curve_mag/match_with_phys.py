import os
from astropy.table import Table
import numpy as np

cat_stack_dir = '/home/lejay/projects/def-sawicki/lejay/completeness_output_mock_cats/rand_pos/'
for cat_name in ['COSMOS_deep', 'DEEP_deep', 'ELAIS_deep', 'XMM-LSS_deep']:
    cat_stack_name = 'cat_stack_'+cat_name+'.fits'
    matched_cat_name = cat_stack_dir+'matched_new_'+cat_stack_name.replace('.fits', '_gal_cut_params.fits')

    cat_stack = Table.read(cat_stack_dir+cat_stack_name)
    print(cat_stack.info)

    # match the completeness estimation catalog with physical catalog
    cat_phys = '/home/lejay/catalogs/v8_cats/'+cat_name+'_v8_gal_cut_params.fits'  # physical catalog but with only useful parameters
    cmd = 'java -jar -Xms128m -Xmx2048m stilts.jar tmatch2 in1='+cat_stack_dir+cat_stack_name + \
     ' in2='+cat_phys+' find=best1 join=all1 matcher=sky params=1 values1="RA_deShift DEC_deShift"' + \
     ' values2="RA'+' DEC'+'" out='+matched_cat_name
    os.system(cmd)

    print(cmd)
    print('matching with physical catalog successful')

    # keep useful parameters
    cat = Table.read(matched_cat_name)
    print(cat.info)
    useful_params = ['ID', 'RA', 'DEC', 'TRACT_insert', 'PATCH_insert', 'FLUX_APER_1.0', 'i', 'ZPHOT', 'Z_BEST', 'MASS_MED',
                     'SSFR_MED', 'sfProb_nuvrz', 'sfProb_nuvrk','sfq_nuvrk','sfq_nuvrz','MASK', 'inside_u', 'inside_uS', 'OBJ_TYPE', 'ORIGINAL']
    cat = cat[useful_params]

    cat = cat[cat['MASK'] == 0]
    if cat_name == 'XMM-LSS_deep':
        cat = cat[cat['inside_uS'] == True]
    else:
        cat = cat[cat['inside_u'] == True]

    cat = cat[~np.isnan(cat['Z_BEST'])]
    cat_gal = cat[cat['OBJ_TYPE'] == 0]
    cat_gal.write(matched_cat_name, overwrite=True)


