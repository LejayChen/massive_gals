from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from astropy.cosmology import WMAP9
import numpy as np
import matplotlib.pyplot as plt
import tqdm

z = 0.6
print(z)
all_gals = 0
sf_gals = 0
q_gals = 0
detected_gals = 0
detected_sfs = 0
detected_qs = 0
massive_count = 0

cat_name = 'SXDS_uddd'
mock_cat = Table.read('matched_cat_stack.fits')
mock_cat = mock_cat[mock_cat['ORIGINAL']==False]
mock_cat = mock_cat[mock_cat['MASS_MED'] > 8.5]

cat_massive_gal = Table.read('../CUT_deep_catalogs/massive_gal_positions/' + cat_name + '_11.15_'+str(z)+'.positions.fits')
bin_number = 14
bin_edges = 10 ** np.linspace(1.0, 2.845, num=bin_number+1)

gal_ids = open('gal_ims.txt').readlines()
pbar = tqdm.tqdm(total=len(gal_ids))

for i in range(len(gal_ids)):
    gal_ids[i] = gal_ids[i].rstrip()
    gal_ids[i] = gal_ids[i].replace('cutout_', '')
    gal_ids[i] = gal_ids[i].replace('.fits', '')

for gal in cat_massive_gal:
    if gal['ID'] in gal_ids:
        pbar.update(1)
        pbar.set_description(gal['ID'])

        massive_count += 1
        dis = WMAP9.angular_diameter_distance(float(gal['z'])).value

        mock_cat_zslice = mock_cat[abs(mock_cat['zKDEPeak'] - float(gal['z'])) < 5*0.044*(1+z)]
        cat_neighbors = mock_cat_zslice[abs(mock_cat_zslice['X_WORLD'] - gal['ra']) < 0.7 / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[abs(cat_neighbors['Y_WORLD'] - gal['dec']) < 0.7 / dis / np.pi * 180]

        coord_gal = SkyCoord(gal['ra'] * u.deg, gal['dec'] * u.deg)
        coord_neighbors = SkyCoord(cat_neighbors['X_WORLD'] * u.deg, cat_neighbors['Y_WORLD'] * u.deg)
        radius_list = coord_neighbors.separation(coord_gal).degree / 180. * np.pi * dis * 1000  # kpc
        radius_list_sf = radius_list[cat_neighbors['SSFR_BEST'] > -11]
        radius_list_q = radius_list[cat_neighbors['SSFR_BEST'] < -11]

        all = np.histogram(radius_list, bins=bin_edges)[0]
        sf = np.histogram(radius_list_sf, bins=bin_edges)[0]
        q = np.histogram(radius_list_q, bins=bin_edges)[0]

        cat_detected = cat_neighbors[~np.isnan(cat_neighbors['FLUX_APER_1.0'])]
        coord_detected = SkyCoord(cat_detected['X_WORLD'] * u.deg, cat_detected['Y_WORLD'] * u.deg)
        radius_list_detected = coord_detected.separation(coord_gal).degree / 180. * np.pi * dis * 1000  # kpc
        radius_list_detected_sf = radius_list_detected[cat_detected['SSFR_BEST'] > -11]
        radius_list_detected_q = radius_list_detected[cat_detected['SSFR_BEST'] < -11]

        detected = np.histogram(radius_list_detected, bins=bin_edges)[0]
        detected_sf = np.histogram(radius_list_detected_sf, bins=bin_edges)[0]
        detected_q = np.histogram(radius_list_detected_q, bins=bin_edges)[0]

        all_gals += all
        sf_gals += sf
        q_gals += q
        detected_gals += detected
        detected_sfs += detected_sf
        detected_qs += detected_q


print(detected_gals)
print(all_gals)
print(detected_sfs)
print(detected_qs)

np.save('completeness.npy', detected_gals/all_gals)
np.save('completeness_sf.npy', detected_sfs/sf_gals)
np.save('completeness_q.npy', detected_qs/q_gals)
np.save('bin_edges.npy', bin_edges)

plt.plot(bin_edges[:-1], detected_gals/all_gals, 'k')
plt.plot(bin_edges[:-1], detected_sfs/sf_gals, 'b')
plt.plot(bin_edges[:-1], detected_qs/q_gals, 'r')
plt.xscale('log')
plt.ylim([0, 1])
# plt.show()
plt.savefig('../figures/spatial_completeness_new.png')