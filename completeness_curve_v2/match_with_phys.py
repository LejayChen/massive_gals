import os
import numpy as np
from astropy.table import Table

output_path = 'Output_cats/'
for cat_name in ['COSMOS_deep', 'DEEP_deep', 'ELAIS_deep', 'XMM-LSS_deep']:
    for z in [0.4, 0.6, 0.8]:
        cat_stack_name = 'cat_stack_'+cat_name+'_'+str(z)+'.fits'
        matched_cat_name = output_path+'phys_matched_new_magaper_100kpc_'+cat_stack_name

        # match the mock galaxy catalog with physical catalog
        cat_phys = '/home/lejay/catalogs/v8_cats/'+cat_name+'_v8_gal_cut_params.fits'
        cmd = 'java -jar -Xms128m -Xmx1500m stilts.jar tmatch2 in1='+output_path + cat_stack_name + \
            ' in2='+cat_phys+' find=best1 join=all1 matcher=sky params=1 values1="RA_deShift DEC_deShift"' + \
            ' values2="RA'+' DEC'+'" out='+matched_cat_name
        print(cmd)
        os.system(cmd)

        # load catalog & cleaning
        cat = Table.read(matched_cat_name)
        cat = cat[cat['MASK'] == 0]  # not masked
        if cat_name == 'XMM-LSS_deep':
            cat = cat[cat['inside_uS'] == True]  # inside u geometry
        else:
            cat = cat[cat['inside_u'] == True]
        cat = cat[~np.isnan(cat['Z_BEST'])]
        cat_gal = cat[cat['OBJ_TYPE'] == 0]  # gal classification

        # keep useful parameters
        useful_params = ['ID', 'RA', 'DEC', 'X_WORLD', 'Y_WORLD', 'MAG_APER', 'FLUX_APER_1.0', 'i', 'ZPHOT', 'Z_BEST',
                         'MASS_MED', 'SSFR_MED', 'sfProb_nuvrz', 'sfProb_nuvrk', 'sfq_nuvrk', 'sfq_nuvrz', 'ORIGINAL']
        cat_gal = cat_gal[useful_params]
        cat_gal.write(matched_cat_name, overwrite=True)

