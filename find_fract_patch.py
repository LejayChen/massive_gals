from astropy.table import Table
from astropy.io import fits
from cutout import *
from astropy import wcs
from astropy.cosmology import WMAP9
import matplotlib.pyplot as plt
import numpy as np
import aplpy

cat = Table.read('CUT2_CLAUDS_HSC_VISTA_Ks23.3_PHYSPARAM_TM.fits')
cat_gal = cat[cat['CLASS'] == 0]
cat_massive_gal = cat_gal[cat_gal['MASS_BEST'] > 11.3]


def find_patch(ra, dec, field='COSMOS'):

    '''find tract and patch by ra,dec'''

    cat_patches = Table.read('tracts_patches/'+field+'_patches.fits')
    for patch in cat_patches:
        if ra < patch['corner0_ra'] and ra > patch['corner1_ra'] and dec < patch['corner2_dec'] and dec > patch['corner1_dec']:
            return str(patch['patch'])[0:-2], str(patch['patch'])[-2:]

    return 0, 0

# main function
for z in np.arange(0.3, 0.31, 0.1):
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

    for gal in cat_massive_z_slice:
        if gal['RA'] > 100:
            tract, patch = find_patch(gal['RA'], gal['DEC'])
        else:
            tract, patch = find_patch(gal['RA'], gal['DEC'], field='XMM-LSS')

        print(gal['ID'], gal['RA'], gal['DEC'], tract, patch)

        if tract != 0:
            try:
                image = fits.open('/media/lejay/Elements/HSC-Z/'+tract+'/'+patch[0]+','+patch[1]+'/'+'calexp-HSC-Z-'+tract+'-'+patch[0]+','+patch[1]+'.fits')
                w = wcs.WCS(image[1].header)
                x, y = w.wcs_world2pix(gal['RA'], gal['DEC'], 0)
                cutoutimg('/media/lejay/Elements/HSC-Z/'+tract+'/'+patch[0]+','+patch[1]+'/'+'calexp-HSC-Z-'+tract+'-'+patch[0]+','+patch[1]+'.fits', gal['RA'], gal['DEC'], xw=10./3600, yw=10./3600, units='wcs', outfile='cutout_images/'+str(z)+'/cutout-'+tract+'-'+patch[0]+','+patch[1]+'_'+str(gal['ID'])+'.fits')

            except FileNotFoundError:
                print('image file for '+tract+' '+patch+' not found!')
                continue

            plt.rc('font', family='serif'), plt.rc('xtick', labelsize=12), plt.rc('ytick', labelsize=12)
            fig = aplpy.FITSFigure('cutout_images/'+str(z)+'/cutout-'+tract+'-'+patch[0]+','+patch[1]+'_'+str(gal['ID'])+'.fits')
            fig.show_grayscale(stretch='arcsinh')
            fig.add_scalebar(0.02/dis/np.pi*180, label='20 kpc')
            fig.show_contour(levels=10,colors='y',alpha=0.4)
            fig.add_label(0.1, 0.95, 'ID='+str(gal['ID']), relative=True, fontsize=12)
            fig.add_label(0.1, 0.9, 'Z=' + str(gal['ZPHOT']), relative=True, fontsize=12)
            if gal['SSFR_BEST'] > -11:
                fig.add_label(0.1, 0.85, 'Star Forming', relative=True, fontsize=12, color='b')
            else:
                fig.add_label(0.1, 0.85, 'Quiescent', relative=True, fontsize=12, color='r')
            fig.add_label(0.11, 0.8, 'log(M)=' + str(round(gal['MASS_BEST'],2)), relative=True, fontsize=12)
            fig.set_theme('publication')
            fig.save('cutout_images/'+str(z)+'/cutout-'+tract+'-'+patch[0]+','+patch[1]+'_'+str(gal['ID'])+'.png')
            fig.close()