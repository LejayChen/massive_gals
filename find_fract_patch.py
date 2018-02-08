from astropy.table import Table
from astropy.io import fits
from cutout import *
from astropy import wcs
import matplotlib.pyplot as plt
from astropy.nddata import Cutout2D

cat = Table.read('CUT_CLAUDS_HSC_VISTA_Ks23.3_PHYSPARAM_TM.fits')
cat_gal = cat[cat['CLASS'] == 0]
cat_massive_gal = cat_gal[cat_gal['MASS_BEST'] > 11.3]
cat_gal_1416 = cat_massive_gal[abs(cat_massive_gal['ZPHOT']-0.5) < 0.1]


def find_patch(ra, dec, field='COSMOS'):

    '''find tract and patch by ra,dec'''

    cat_patches = Table.read('tracts_patches/'+field+'_patches.fits')
    for patch in cat_patches:
        if ra < patch['corner0_ra'] and ra > patch['corner1_ra'] and \
           dec < patch['corner2_dec'] and dec > patch['corner1_dec']:

            return str(patch['patch'])[0:-2], str(patch['patch'])[-2:]

    return 0, 0

for gal in cat_gal_1416:
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
            cutout = Cutout2D(image[1].data, (x, y), (100, 100), mode='partial', wcs=w)
            plt.imshow(cutout.data, origin='lower')
            plt.title(str(round(gal['RA'],5))+' '+str(round(gal['DEC'],5))+' '+str(gal['z']))
            plt.savefig('cutout_images/low_z/cutout-'+tract+'-'+patch[0]+','+patch[1]+'_'+str(gal['ID'])+'.png')

            cutoutimg('/media/lejay/Elements/HSC-Z/'+tract+'/'+patch[0]+','+patch[1]+'/'+'calexp-HSC-Z-'+tract+'-'+patch[0]+','+patch[1]+'.fits', gal['RA'], gal['DEC'], xw=10./3600, yw=10./3600, units='wcs', outfile='cutout_images/low_z/cutout-'+tract+'-'+patch[0]+','+patch[1]+'_'+str(gal['ID'])+'.fits')

        except FileNotFoundError:
            print('image file for '+tract+' '+patch+' not found!')
            continue














































