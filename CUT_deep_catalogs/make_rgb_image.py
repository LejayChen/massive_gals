from cutout import *
import numpy as np
import aplpy, os
from astropy.cosmology import WMAP9
from astropy.table import Table
from astropy.io import fits


def download_image_single(tract, patch,band):
    tract = str(tract)
    patch = str(patch)
    img = 'calexp-HSC-'+str(band)+'-'+tract+'-'+patch+'.fits'
    os.system('vcp vos:clauds/coupon/s16a_udeep_deep_depth.ext_v2.0/deepCoadd/HSC-'+band+'/'
              + tract + '/' + patch + '/' + img + ' ./ --verbose')

    return img


def download_image(tract, patch):
    tract = str(tract)
    patch = str(patch)
    img_r = 'calexp-HSC-R-'+tract+'-'+patch+'.fits'
    img_g = 'calexp-HSC-G-'+tract+'-'+patch+'.fits'
    img_b = 'calexp-MegaCam-uS-'+tract+'-'+patch+'.fits'

    os.system('vcp vos:clauds/coupon/s16a_udeep_deep_depth.ext_v2.0/deepCoadd/HSC-R/'
              +tract+'/'+patch + '/' + img_r+' ./ --verbose')
    os.system('vcp vos:clauds/coupon/s16a_udeep_deep_depth.ext_v2.0/deepCoadd/HSC-G/'
              +tract+'/'+patch + '/' + img_g + ' ./ --verbose')
    os.system('vcp vos:clauds/coupon/s16a_udeep_deep_depth.ext_v2.0/deepCoadd/MegaCam-uS/'
              +tract+'/'+patch + '/' + img_b + ' ./ --verbose')
    return img_r, img_g, img_b


def find_patch(ra, dec, field='XMM-LSS'):
    cat_patches = Table.read('../tracts_patches/'+field+'_patches.fits')
    for patch in cat_patches:
        if ra < patch['corner0_ra']-0.09 and ra > patch['corner1_ra']+0.09 and dec < patch['corner2_dec']-0.09 and dec > patch['corner1_dec']+0.09:
            return str(patch['patch'])[0:-2], str(patch['patch'])[-2:]
    return 0, 0


def cut_rgb_image(tract, patch, ra, dec):
    width = 10/3600.0  # in degree
    img_r, img_g, img_b = download_image(tract, patch)

    try:
        cutoutimg(img_r, ra, dec, xw=width, yw=width, units='wcs', outfile='central_r.fits',
           overwrite=True, useMontage=False, coordsys='celestial', verbose=False, centerunits=None)
        cutoutimg(img_g, ra, dec, xw=width, yw=width, units='wcs', outfile='central_g.fits',
           overwrite=True, useMontage=False, coordsys='celestial', verbose=False, centerunits=None)
        cutoutimg(img_b, ra, dec, xw=width, yw=width, units='wcs', outfile='central_us.fits',
           overwrite=True, useMontage=False, coordsys='celestial', verbose=False, centerunits=None)
    except:
        os.system('rm ' + img_r + ' ' + img_g + ' ' + img_b)
        return 'no output img', False

    output_img = 'rgb_image.png'
    aplpy.make_rgb_cube(['central_r.fits', 'central_g.fits', 'central_us.fits'], 'cube.fits')
    aplpy.make_rgb_image('cube.fits', output_img,
                         stretch_r='log', stretch_g='log',
                         stretch_b='log', vmin_r=0, vmin_g=0, vmin_b=0)

    os.system('rm '+img_r+' '+img_g+' '+img_b)
    os.system('rm cube.fits')
    return output_img, True


def cut_i_band_img(tract, patch, ra, dec, band):
    width = 6 / 3600.0  # in degree
    img = download_image_single(tract, patch,'I')

    try:
        cutoutimg(img, ra, dec, xw=width, yw=width, units='wcs', outfile='central_'+str(band)+'.fits',
                  overwrite=True, useMontage=False, coordsys='celestial', verbose=False, centerunits=None)
    except:
        os.system('rm ' + img)
        return 'no output img', False

    # output_img = str(band)+'_image.png'
    # img = aplpy.FITSFigure('central_'+str(band)+'.fits')
    # img.show_grayscale()
    # img.remove_grid()
    # img.save(output_img)
    #
    # os.system('rm ' + 'central_'+str(band)+'.fits')
    return 'central_'+str(band)+'.fits', True


if __name__ == '__main__':
    cat = Table(fits.getdata('CUT3_XMM-LSS_deep.fits'))
    cat = cat[cat['zKDEPeak'] < 1]
    cat_gal = cat[np.logical_and(cat['preds_median'] < 0.89, cat['inside'] == True)]
    cat_gal = cat_gal[cat_gal['MASS_MED'] > 9.0]
    gal = cat_gal[cat_gal['NUMBER'] == 8524650006521][0]
    dis = WMAP9.angular_diameter_distance(gal['zKDEPeak']).value

    cut_rgb_image(gal['TRACT'], gal['PATCH'], gal['RA'], gal['DEC'])