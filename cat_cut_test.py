from astropy.table import Table
from astropy.cosmology import WMAP9
import time
import numpy as np
import astropy.units as u
from random import random
from astropy.coordinates import SkyCoord, match_coordinates_sky
from sklearn.neighbors import KDTree

cat = Table.read('CUT2_CLAUDS_HSC_VISTA_Ks23.3_PHYSPARAM_TM.fits')
cat_gal = cat[cat['CLASS'] == 0]

z = 0.5
dis = WMAP9.angular_diameter_distance(z).value
cat_all_z_slice = cat_gal[abs(cat_gal['ZPHOT'] - z) < 0.5]
ra = 150 + random()*2.0/dis/np.pi*180
dec = 2 + random()*2.0/dis/np.pi*180

start_1 = time.time()
coord_gal = SkyCoord(ra * u.deg, dec * u.deg)
cat_neighbors = cat_all_z_slice[abs(cat_all_z_slice['RA']-ra) < 1.0/dis/np.pi*180]
cat_neighbors = cat_neighbors[abs(cat_neighbors['DEC']-dec) < 1.0/dis/np.pi*180]
coord_neighbors = SkyCoord(cat_neighbors['RA']*u.deg, cat_neighbors['DEC']*u.deg)
cat_neighbors = cat_neighbors[coord_neighbors.separation(coord_gal).degree < 1.0 / dis / np.pi * 180]
print(len(cat_neighbors))
print("--- %s seconds ---" % (time.time() - start_1))

start_2 = time.time()
coord_gal = SkyCoord(ra * u.deg, dec * u.deg)
coord_neighbors = SkyCoord(cat_all_z_slice['RA'] * u.deg, cat_all_z_slice['DEC'] * u.deg)
cat_neighbors = cat_all_z_slice[coord_neighbors.separation(coord_gal).degree < 1.0 / dis / np.pi * 180]
print(len(cat_neighbors))
print("--- %s seconds ---" % (time.time() - start_2))

start_3 = time.time()
cat_neighbors = cat_all_z_slice[abs(cat_all_z_slice['RA']-ra) < 1.0/dis/np.pi*180]
cat_neighbors = cat_neighbors[abs(cat_neighbors['DEC']-dec) < 1.0/dis/np.pi*180]
tree = KDTree(np.array(cat_neighbors['RA', 'DEC']).tolist())
ind = tree.query_radius([(ra, dec)], 1.0 / dis / np.pi * 180)
cat_neighbors = cat_all_z_slice[ind[0]]
print(len(cat_neighbors))
print("--- %s seconds ---" % (time.time() - start_3))