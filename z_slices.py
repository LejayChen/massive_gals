from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np

cat = Table.read('CLAUDS_HSC_VISTA_Ks23.3_PHYSPARAM_TM.fits')
cat_gal = cat[cat['CLASS'] == 0]
cat_massive_gal = cat_gal[cat_gal['MASS_BEST'] > 11.3]

for z in np.arange(1.1, 2.5, 0.1):
    cat_z_slice = cat_massive_gal[abs(cat_massive_gal['ZPHOT']-z)<0.1]
    cat_sf = cat_z_slice[cat_z_slice['SSFR_BEST'] > -11]
    cat_qs = cat_z_slice[cat_z_slice['SSFR_BEST'] < -11]
    fig = plt.figure(figsize=(9, 9))
    plt.rc('font', family='serif'), plt.rc('xtick', labelsize=15), plt.rc('ytick', labelsize=15)

    ax = fig.add_subplot(111)
    ax.plot(cat['RA'], cat['DEC'], '.', color='k', alpha=0.01)
    ax.plot(cat_sf['RA'], cat_sf['DEC'], '.', color='b', label='Star Forming')
    ax.plot(cat_qs['RA'], cat_qs['DEC'], '.', color='r', label='Quiescent')

    ax.set_xlabel('R.A. (degrees)', fontsize=16)
    ax.set_ylabel('Dec. (degrees)', fontsize=16)
    ax.set_title('in COSMOS field', fontsize=16)
    ax.text(149.2, 1.3, 'z='+str(z-0.1)+'~'+str(z+0.1), fontsize=16)

    ax.axis([149.1, 151, 1.2, 3.1])
    ax.grid(True)
    ax.legend(fontsize=16, loc='lower right')

    plt.savefig('z_slices/COSMOS_'+str(z-0.1)+'_'+str(z+0.1)+'.png')

    print(z)
