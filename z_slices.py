from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9
import numpy as np

cat = Table.read('CUT2_CLAUDS_HSC_VISTA_Ks23.3_PHYSPARAM_TM.fits')
cat_gal = cat[cat['CLASS'] == 0]
cat_massive_gal = cat_gal[cat_gal['MASS_BEST'] > 11.0]

for z in np.arange(0.3, 2.5, 0.1):
    print('============='+str(z)+'================')
    dis = WMAP9.angular_diameter_distance(z).value  # angular diameter distance at redshift z
    dis_l = WMAP9.comoving_distance(z - 0.1).value  # comoving distance at redshift z-0.1
    dis_h = WMAP9.comoving_distance(z + 0.1).value  # comoving distance at redshift z+0.1
    total_v = 4. / 3 * np.pi * (dis_h ** 3 - dis_l ** 3)  # Mpc^3
    survey_v = total_v * (4 / 41253.05)  # Mpc^3
    density = 0.00003  # desired constant (cumulative) volume number density (Mpc^-3)
    num = int(density * survey_v)

    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal['ZPHOT'] - z) < 0.1]  # massive galaxies in this z slice
    cat_massive_z_slice.sort('MASS_BEST')
    cat_massive_z_slice.reverse()
    cat_massive_z_slice = cat_massive_z_slice[:num]
    print(num)

    cat_sf = cat_massive_z_slice[cat_massive_z_slice['SSFR_BEST'] > -11]
    cat_qs = cat_massive_z_slice[cat_massive_z_slice['SSFR_BEST'] < -11]
    fig = plt.figure(figsize=(9, 9))
    plt.rc('font', family='serif'), plt.rc('xtick', labelsize=15), plt.rc('ytick', labelsize=15)

    ax = fig.add_subplot(111)
    ax.plot(cat['RA'], cat['DEC'], '.', color='k', alpha=0.01)
    ax.plot(cat_sf['RA'], cat_sf['DEC'], '.', color='b', label='Star Forming')
    ax.plot(cat_qs['RA'], cat_qs['DEC'], '.', color='r', label='Quiescent')

    ax.set_xlabel('R.A. (degrees)', fontsize=16)
    ax.set_ylabel('Dec. (degrees)', fontsize=16)
    # ax.set_title('in COSMOS field', fontsize=16)
    ax.set_title('in XMM-LSS field', fontsize=16)

    ax.axis([33.75, 35.75, -6, -4])
    ax.text(33.9, -5.9, 'z=' + str(z - 0.1) + '~' + str(z + 0.1), fontsize=16)
    # ax.axis([149.1, 151, 1.2, 3.1])
    # ax.text(149.2, 1.3, 'z=' + str(z - 0.1) + '~' + str(z + 0.1), fontsize=16)

    ax.grid(True)
    ax.legend(fontsize=16, loc='lower right')

    plt.savefig('z_slices/XMM_LSS'+str(z - 0.1)+'_'+str(z + 0.1)+'.png')
    # plt.savefig('z_slices/COSMOS_'+str(z-0.1)+'_'+str(z+0.1)+'.png')

    print('z =', z)