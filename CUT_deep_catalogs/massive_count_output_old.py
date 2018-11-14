from astropy.table import *
from astropy.io import fits
import sys
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import WMAP9
from sklearn.neighbors import KDTree

path = 'massive_counts_inside/'
masscut = 11.15
cat_name = sys.argv[1]
z_keyname = 'zKDEPeak'
mass_keyname = 'MASS_MED'
z_bin_size = sys.argv[2]

print('start reading catalog', end='\r')
cat = Table(fits.getdata('/home/lejay/s16a_'+cat_name+'_masterCat.fits'))
cat = cat[cat[z_keyname] < 1]
cat_gal = cat[np.logical_and(cat['preds_median'] < 0.89, cat['inside']==True)]
cat_gal = cat_gal[cat_gal[mass_keyname] > 9.0]
cat_massive_gal = cat_gal[cat_gal[mass_keyname] > masscut]

cat_massive_counts = Table(names=('zphot', 'count', 'count_q', 'count_sf'))
for z in np.array([0.6]):
    cat_massive_gal_positions = Table(names=('ID', 'z', 'ssfr', 'sfprob', 'mass', 'ra', 'dec', 'tract', 'patch'), dtype=('S16', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'S16', 'S16'))
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal['zKDEPeak'] - z) < eval(z_bin_size)]
    coord_massive_gal = SkyCoord(cat_massive_z_slice['RA'] * u.deg, cat_massive_z_slice['DEC'] * u.deg)
    cat_all_z_slice = cat_gal[abs(cat_gal[z_keyname] - z) < 0.5]

    massive_counts = len(cat_massive_z_slice)
    massive_count = 0
    massive_counts_cq = 0
    massive_counts_csf = 0
    for gal in cat_massive_z_slice:
        massive_count += 1
        print('Progress:' + str(massive_count) + '/' + str(len(cat_massive_z_slice)), end='\r')
        dis = WMAP9.angular_diameter_distance(gal[z_keyname]).value
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)
        cat_neighbors_z_slice = cat_all_z_slice[abs(cat_all_z_slice[z_keyname] - gal[z_keyname]) < 1.5 * 0.044 * (1 + z)]
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
        if gal[mass_keyname] < max(cat_neighbors[mass_keyname]):  # no more-massive companions
            massive_counts -= 1
            continue
        cat_massive_gal_positions.add_row([gal['NUMBER'], gal['zKDEPeak'], gal['SSFR_BEST'], gal['sfProb'], gal['MASS_MED'], gal['RA'], gal['DEC'], gal['TRACT'], gal['PATCH']])

        # cut on central SF/Q
        if gal['SSFR_BEST'] < -11:
            massive_counts_cq += 1
        else:
            massive_counts_csf += 1

    cat_massive_counts.add_row([z, massive_counts, massive_counts_cq, massive_counts_csf])
    cat_massive_gal_positions.write('massive_gal_positions/isolated_'+cat_name+'_'+str(masscut)+'_'+str(z)+'.positions.fits', overwrite=True)
    print('                                            ', end='\r')
    print(z, massive_counts, massive_counts_csf, massive_counts_cq)

# cat_massive_counts.write(path + cat_name +'_'+str(z_bin_size)+'_'+str(masscut)+'.fits', overwrite=True)