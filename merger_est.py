from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import WMAP9
from astropy.coordinates import SkyCoord
import astropy.units as u

cat = Table.read('CUT2_CLAUDS_HSC_VISTA_Ks23.3_PHYSPARAM_TM.fits')
cat_gal = cat[cat['CLASS'] == 0]
cat_massive_gal = cat_gal[cat_gal['MASS_BEST'] > 11.0]


def merge_est(r, m, z):
    m = 10**(m - 10)
    t_merge = 3.2*(r/50.)*(m/4)**(-0.3)*(1+z/20.)
    return np.array(t_merge)


def merge_est2(r, m_sat, m_cen):
    m_sat = 10**(m_sat - 10)
    m_cen = 10 ** (m_cen - 10)
    t_merge = (0.94*0.5**0.6+0.6)/0.86*m_cen/m_sat*1/(1+m_cen/m_sat)*r/860.
    return np.array(t_merge)


fig = plt.figure(figsize=(10, 8))
plt.rc('font', family='serif'), plt.rc('xtick', labelsize=16), plt.rc('ytick', labelsize=16)


mass_growth_file = open('mass_growth','w')
for z in np.arange(0.3, 1.91, 0.1):
    print('=============' + str(z) + '================')

    dis = WMAP9.angular_diameter_distance(z).value  # angular diameter distance at redshift z
    dis_l = WMAP9.comoving_distance(z - 0.1).value  # comoving distance at redshift z-0.1
    dis_h = WMAP9.comoving_distance(z + 0.1).value  # comoving distance at redshift z+0.1
    total_v = 4. / 3 * np.pi * (dis_h ** 3 - dis_l ** 3)  # Mpc^3
    survey_v = total_v * 4 / 41253.05  # Mpc^3
    density = 0.00003  # desired constant (cumulative) volume number density (Mpc^-3)
    num = int(density * survey_v)

    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal['ZPHOT'] - z) < 0.1]  # massive galaxies in this z slice
    cat_massive_z_slice.sort('MASS_BEST')
    cat_massive_z_slice.reverse()
    cat_massive_z_slice = cat_massive_z_slice[:num]  # select most massive ones (keep surface density constant in different redshift bins)

    cat_all_z_slice = cat_gal[abs(cat_gal['ZPHOT'] - z) < 0.1]


    mass_growth_gal = []
    for gal in cat_massive_z_slice:
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)
        coord_all_z_slice = SkyCoord(cat_all_z_slice['RA'] * u.deg, cat_all_z_slice['DEC'] * u.deg)
        cat_neighbors = cat_all_z_slice[coord_all_z_slice.separation(coord_gal).degree < 0.05 / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[cat_neighbors['ID'] != gal['ID']]
        cat_neighbors = cat_neighbors[cat_neighbors['MASS_BEST']>9]


        if len(cat_neighbors) == 0:  # exlucde central gals which has no companion
            continue

        if gal['MASS_BEST'] < max(cat_neighbors['MASS_BEST']):  # exclude central gals which has larger mass companion
            continue

        # cat_neighbors = cat_neighbors[cat_neighbors['MASS_BEST']>8]

        coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
        radius_list = coord_neighbors.separation(coord_gal).degree/180.*np.pi*dis*1000  # in kpc
        mass_list = cat_neighbors['MASS_BEST']

        # t_merge_list = merge_est(radius_list, mass_list, z)
        t_merge_list = merge_est2(radius_list, mass_list, gal['MASS_BEST'])

        # select companions that merge with central in the next redshift bin
        t_merge_h = WMAP9.lookback_time(gal['ZPHOT']).value - WMAP9.lookback_time(z - 0.2).value
        t_merge_l = WMAP9.lookback_time(gal['ZPHOT']).value - WMAP9.lookback_time(z).value
        mass_list = mass_list[abs(t_merge_list-(t_merge_h+t_merge_l)/2.)< (t_merge_h-t_merge_l)/2.]

        mass_growth_gal.append(sum(10**(mass_list-10)))

    mass_growth_z_median = np.median(mass_growth_gal)
    mass_growth_z_mean = np.mean(mass_growth_gal)
    mass_growth_file.write(str(mass_growth_z_mean)+'\n')
    print(mass_growth_z_mean)

mass_growth_file.close()