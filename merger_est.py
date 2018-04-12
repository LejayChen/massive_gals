# output mass_growth file

from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import WMAP9
from astropy.coordinates import SkyCoord
from random import random
import astropy.units as u

cat = Table.read('CUT2_CLAUDS_HSC_VISTA_Ks23.3_PHYSPARAM_TM.fits')
cat_gal = cat[cat['CLASS'] == 0]
cat_gal = cat_gal[cat_gal['MASS_BEST'] > 9.5]
cat_massive_gal = cat_gal[cat_gal['MASS_BEST'] > 11.0]


def bkg(ra_central, cat_all_z_slice_rand, t_merge_h_rand, t_merge_l_rand, z, m_central):
    n = 0
    num_p = 10
    mass_z_rand = []
    while n < num_p:  # get several blank pointing's to estimate background
        same_field = False  # find a random pointing in the same field as the central galaxy
        while not same_field:
            id_rand = int(random()*len(cat_all_z_slice_rand))
            ra_rand = cat_all_z_slice_rand[id_rand]['RA']+ random()*1.5/dis/np.pi*180
            dec_rand = cat_all_z_slice_rand[id_rand]['DEC']+ random()*1.5/dis/np.pi*180
            if ra_rand > 100 and ra_central > 100:
                same_field = True
            elif ra_rand < 100 and ra_central < 100:
                same_field = True
            else:
                same_field = False

        # construct neighbors catalog
        coord_all_z_slice_rand = SkyCoord(cat_all_z_slice_rand['RA'] * u.deg, cat_all_z_slice_rand['DEC'] * u.deg)
        cat_neighbors_rand = cat_all_z_slice_rand[coord_all_z_slice_rand.separation(SkyCoord(ra_rand*u.deg, dec_rand*u.deg)).degree < 0.5/dis/np.pi*180.]
        # cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['ID'] != id_rand]
        # cat_neighbors_rand = cat_neighbors_rand[cat_neighbors_rand['MASS_BEST'] > 9]

        coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)
        radius_list_rand = coord_neighbors_rand.separation(SkyCoord(ra_rand*u.deg, dec_rand*u.deg)).degree/180.*np.pi*dis*1000  # kpc
        mass_list_rand = cat_neighbors_rand['MASS_BEST']

        t_merge_list_rand = merge_est_k(radius_list_rand, m_central, z)
        # t_merge_list_rand = merge_est_j(radius_list_rand, mass_list_rand, m_central)

        mass_list_rand = mass_list_rand[np.absolute(t_merge_list_rand - (t_merge_h_rand + t_merge_l_rand) / 2.) < (t_merge_h_rand - t_merge_l_rand) / 2.]
        mass_z_rand.append(sum(10**(mass_list_rand-10)))

        n = n + 1

    return np.mean(mass_z_rand)


def merge_est_k(r, m_cen, z):
    m = 10 ** (m_cen - 10)
    h = 1
    t_merge = 3.2*(r/50.) * (m/4*h)**(-0.3) * (1+z/20.)  # Gyr

    return np.array(t_merge)


def merge_est_km(r, m_cen, z):
    if r < 0.03:
        t0 = 2806
        f1 = -94.7 * 1e-5
        f2 = 671 * 1e-5
    elif r < 0.05:
        t0 = 4971
        f1 = -38.6 * 1e-5
        f2 = 615 * 1e-5
    else:
        t0 = 11412
        f1 = 18 * 1e-5
        f2 = 491 * 1e-5

    t_merge = (t0**(-0.5)+f1*z+f2*(m_cen-10))**(-2)

    return t_merge


def merge_est_j(r, m_sat, m_cen):
    m_sat = 10 ** (m_sat - 10)
    m_cen = 10 ** (m_cen - 10)
    r = r*3.086e16  # kpc to km
    t_merge = (0.94*0.5**0.6+0.6)/0.86 * (m_cen/m_sat) * (1/np.log(1+m_cen/m_sat)) * r/860.  # second
    t_merge = t_merge/(365.25*24*60*60)/1e9  # second to Gyr

    return np.array(t_merge)


fig = plt.figure(figsize=(10, 8))
plt.rc('font', family='serif'), plt.rc('xtick', labelsize=16), plt.rc('ytick', labelsize=16)
mass_growth_file = open('mass_growth', 'w')
for z in np.arange(3, 20)/10.:
    print('=============' + str(z) + '================')

    dis = WMAP9.angular_diameter_distance(z).value  # angular diameter distance at redshift z (Mpc)
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
        cat_neighbors = cat_all_z_slice[coord_all_z_slice.separation(coord_gal).degree < 0.5 / dis / np.pi * 180]
        cat_neighbors = cat_neighbors[cat_neighbors['ID'] != gal['ID']]
        # cat_neighbors = cat_neighbors[cat_neighbors['MASS_BEST'] > 9]

        if len(cat_neighbors) == 0:  # exclude central gals which has no companion
            continue

        if gal['MASS_BEST'] < max(cat_neighbors['MASS_BEST']):  # exclude central gals which has larger mass companion
            continue

        # cat_neighbors = cat_neighbors[cat_neighbors['MASS_BEST']>8]

        coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
        radius_list = coord_neighbors.separation(coord_gal).degree/180.*np.pi*dis*1000  # in kpc
        mass_list = cat_neighbors['MASS_BEST']

        t_merge_list = merge_est_k(radius_list, mass_list, z)
        # t_merge_list = merge_est_j(radius_list, mass_list, gal['MASS_BEST'])

        # select companions that merge with central in the next redshift bin
        t_merge_h = WMAP9.lookback_time(gal['ZPHOT']).value - WMAP9.lookback_time(z - 0.2).value
        t_merge_l = WMAP9.lookback_time(gal['ZPHOT']).value - WMAP9.lookback_time(z).value
        mass_grow_list = mass_list[np.absolute(t_merge_list-(t_merge_h+t_merge_l)/2.) < (t_merge_h-t_merge_l)/2.]

        mass_growth_bkg = bkg(gal['RA'], cat_all_z_slice, t_merge_h, t_merge_l, z, gal['MASS_BEST'])
        mass_growth_gal.append(sum(10**(mass_grow_list-10)) - mass_growth_bkg)

        print(len(mass_grow_list), len(mass_list),sum(10**(mass_grow_list-10)),mass_growth_bkg)

    mass_growth_z_median = np.median(mass_growth_gal)
    mass_growth_z_mean = np.mean(mass_growth_gal)
    mass_growth_file.write(str(mass_growth_z_mean)+'\n')
    print(mass_growth_z_mean)

mass_growth_file.close()