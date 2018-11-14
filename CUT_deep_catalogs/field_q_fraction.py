from astropy.table import Table
import numpy as np

q_frac = []
for z in [0.2, 0.4, 0.6, 0.8]:
	n_sf = 0
	n_q = 0
	for cat_name in ['COSMOS_deep','ELAIS_deep','XMM-LSS_deep','DEEP_deep','SXDS_uddd']:
		print(z,cat_name)
		cat = Table.read('CUT3_'+cat_name+'.fits')
		cat = cat[cat['zKDEPeak'] < 1]
		cat_gal = cat[np.logical_and(cat['preds_median'] < 0.89, cat['inside'] == True)]
		cat_z_slice = cat_gal[abs(cat_gal['zKDEPeak']-z)>0.1]
		cat_mass_slice = cat_z_slice[cat_z_slice['MASS_MED']>9.5]

		n_sf += np.sum(cat_mass_slice['sfProb'])
		n_q += np.sum(1-cat_mass_slice['sfProb'])

	q_frac.append(n_q/(n_sf+n_q))
	print(n_q/(n_sf+n_q))
	
q_frac = np.array(q_frac)
np.savetxt('field_q_frac.txt',q_frac)