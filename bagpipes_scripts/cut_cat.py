from astropy.table import Table
import numpy as np
import sys


n_gals = eval(sys.argv[1])
cat_name = sys.argv[2]
np.random.seed(101)

cat = Table.read('../CUT_deep_catalogs/'+cat_name+'_photoz_v2.fits')

# for central cosmos region
cat = cat[cat['RA'] > 149.4]
cat = cat[cat['RA'] < 150.8]
cat = cat[cat['DEC'] > 1.5]
cat = cat[cat['DEC'] < 2.9]

#cat = cat[np.random.randint(len(cat), size=n_gals)]

# cut out unreliable sources
cat = cat[cat['MAG_APER_2s_u'] < 30.0]
cat = cat[cat['MAG_APER_2s_g'] < 30.0]
cat = cat[cat['MAG_APER_2s_r'] < 30.0]
cat = cat[cat['MAG_APER_2s_i'] < 30.0]
cat = cat[cat['i'] < 25.0]
cat = cat[cat['MAG_APER_2s_z'] < 30.0]
cat = cat[cat['MAG_APER_2s_y'] < 30.0]

cat = cat[cat['MAG_APER_2s_u'] > 0.0]
cat = cat[cat['MAG_APER_2s_g'] > 0.0]
cat = cat[cat['MAG_APER_2s_r'] > 0.0]
cat = cat[cat['MAG_APER_2s_i'] > 0.0]
cat = cat[cat['i'] > 0.0]
cat = cat[cat['MAG_APER_2s_z'] > 0.0]
cat = cat[cat['MAG_APER_2s_y'] > 0.0]

# SNR > 2.08
cat = cat[cat['MAGERR_APER_2s_u'] < 0.3]
cat = cat[cat['MAGERR_APER_2s_g'] < 0.3]
cat = cat[cat['MAGERR_APER_2s_r'] < 0.3]
cat = cat[cat['MAGERR_APER_2s_i'] < 0.3]
cat = cat[cat['MAGERR_APER_2s_z'] < 0.3]
cat = cat[cat['MAGERR_APER_2s_y'] < 0.3]

# redshift cut
cat = cat[cat['Z_BEST'] < 1.3]
cat = cat[cat['Z_BEST'] > 0]

cat = cat[cat['Z_BEST68_LOW'] > 0]
cat = cat[cat['Z_BEST'] - cat['Z_BEST68_LOW'] > 0.001]
cat = cat[cat['Z_BEST68_HIGH'] - cat['Z_BEST68_LOW'] > 0.002]

# mask cut
cat = cat[cat['inside'] == True]
cat = cat[cat['MASK'] != 0]

if len(cat) != 0:
    print('number of gals to be fitted:', len(cat))
    cat.write('../CUT_deep_catalogs/'+cat_name+'_to_fit.fits', overwrite=True)
else:
    raise ValueError('catalog length must be nonzero, getting '+str(len(cat))+'.')
