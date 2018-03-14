from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import WMAP9
from astropy.coordinates import SkyCoord
import astropy.units as u

cat = Table.read('CUT2_CLAUDS_HSC_VISTA_Ks23.3_PHYSPARAM_TM.fits')
cat_gal = cat[cat['CLASS'] == 0]
cat_massive_gal = cat_gal[cat_gal['MASS_BEST'] > 11.3]


def merge_est(r, m, z):
    m = 10**(m - 10)
    t_merge = 3.2*(r/50.)*(m/4)**(-0.3)*(1+z/20.)
    return np.array(t_merge)


fig = plt.figure(figsize=(10, 8))
plt.rc('font', family='serif'), plt.rc('xtick', labelsize=16), plt.rc('ytick', labelsize=16)

for z in np.arange(0.3, 0.5, 0.1):
    print('=============' + str(z) + '================')
    dis = WMAP9.angular_diameter_distance(z).value  # in Mpc
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal['ZPHOT'] - z) < 0.1]
    cat_all_z_slice = cat_gal[abs(cat_gal['ZPHOT'] - z) < 0.1]

    for gal in cat_massive_z_slice:
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)
        coord_all_z_slice = SkyCoord(cat_all_z_slice['RA'] * u.deg, cat_all_z_slice['DEC'] * u.deg)
        cat_neighbors = cat_all_z_slice[coord_all_z_slice.separation(coord_gal).degree < 0.5 / dis / np.pi * 180]

        coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
        radius_list = coord_neighbors.separation(coord_gal).degree/180.*np.pi*dis*1000  # in kpc
        mass_list = cat_neighbors['MASS_BEST']

        t_merge_list = merge_est(radius_list, mass_list, z)

        sort = t_merge_list.argsort()
        t_merge_list = t_merge_list[sort[::1]]
        mass_list = mass_list[sort[::1]]

        total_gal_mass = 10**(gal['MASS_BEST'] - 10)
        lookback_time_list = []
        gal_mass_list = []
        for i in range(len(t_merge_list)):
            lookback_time_list.append(WMAP9.lookback_time(z).value - t_merge_list[i])
            total_gal_mass += 10**(mass_list[i] - 10)
            gal_mass_list.append(total_gal_mass)

        plt.plot(lookback_time_list, gal_mass_list,alpha=0.5)
        plt.plot(lookback_time_list[0], gal_mass_list[0], 'o', markersize=5)

plt.xlabel('Lookback Time', fontsize=15)
plt.ylabel('Cumulative Mass (M_sun)', fontsize=15)
plt.show()