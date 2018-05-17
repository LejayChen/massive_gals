from astropy.table import *
import sys
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import WMAP9
from sklearn.neighbors import KDTree

bin_size = 0.09
masscut = 11.15
cat_name = sys.argv[1]
cat = Table.read('CUT_'+cat_name+'.fits')
cat = cat[cat['zKDEPeak'] < 1]
cat_gal = cat[cat['preds_median'] < 0.89]
cat_gal = cat_gal[cat_gal['MASS_MED'] > 9.0]
cat_massive_gal = cat_gal[cat_gal['MASS_MED'] > masscut]

cat_massive_counts = Table(names=('zphot', 'count', 'count_q', 'count_sf'))
for z in np.arange(4, 8.1, 1)/10.:
    print('=============' + str(z) + '================')
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal['zKDEPeak'] - z) < bin_size]
    coord_massive_gal = SkyCoord(cat_massive_z_slice['RA'] * u.deg, cat_massive_z_slice['DEC'] * u.deg)
    cat_all_z_slice = cat_gal[abs(cat_gal['zKDEPeak'] - z) < 0.5]

    massive_counts = len(cat_massive_z_slice)
    massive_count = 0
    massive_counts_cq = 0
    massive_counts_csf = 0
    for gal in cat_massive_z_slice:
        massive_count += 1
        print('Progress:' + str(massive_count) + '/' + str(len(cat_massive_z_slice)), end='\r')
        dis = WMAP9.angular_diameter_distance(gal['zKDEPeak']).value
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)
        cat_neighbors_z_slice = cat_all_z_slice[
            abs(cat_all_z_slice['zKDEPeak'] - gal['zKDEPeak']) < 1.5 * 0.03 * (1 + z)]
        cat_neighbors = cat_neighbors_z_slice[abs(cat_neighbors_z_slice['RA'] - gal['RA']) < 0.7 / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[abs(cat_neighbors['DEC'] - gal['DEC']) < 0.7 / dis / np.pi * 180]
        if len(cat_neighbors) == 0:  # central gals which has no companion
            continue
        else:
            ind = KDTree(np.array(cat_neighbors['RA', 'DEC']).tolist()).query_radius([(gal['RA'], gal['DEC'])], 0.7 / dis / np.pi * 180)
            cat_neighbors = cat_neighbors[ind[0]]
            cat_neighbors = cat_neighbors[cat_neighbors['NUMBER'] != gal['NUMBER']]
            if len(cat_neighbors) == 0:  # central gals which has no companion
                continue

        # isolation cut on central
        if gal['MASS_MED'] < max(cat_neighbors['MASS_MED']):  # no more-massive companions
            massive_counts -= 1
            continue

        # cut on central SF/Q
        if gal['SSFR_BEST'] < -11:
            massive_counts_cq += 1
        else:
            massive_counts_csf += 1

    cat_massive_counts.add_row([z, massive_counts, massive_counts_cq, massive_counts_csf])
    print('                                            ', end='\r')
    print(z, massive_counts, massive_counts_csf, massive_counts_cq)

cat_massive_counts.write('massive_counts/'+cat_name+'_'+str(bin_size)+'_'+str(masscut)+'.fits')