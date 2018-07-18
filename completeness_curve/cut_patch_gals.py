from cutout import cutoutimg
from astropy.table import Table
from astropy.io import fits
from astropy import wcs
from random import random
import numpy as np
import os

ims = open('ims.txt').readlines()
gal_ims = open('gal_ims.txt', 'w')
for im in ims:
    im = im.rstrip()
    mask = im.replace('chi2', 'chi2Mask')
    tract = im.split('_')[1]
    patch = im.split('_')[2].replace('.fits', '')
    patch = patch[0]+','+patch[1]

    # maskLoc = "clauds/anneya/s16a_v2/s16a_udddv2.2/SExChi2_chi2Masks/" + mask
    # imLoc = "clauds/anneya/s16a_v2/s16a_udddv2.2/SExChi2_chi2Ims/" + im
    # os.system('vcp vos:' + imLoc + ' ./ --verbose ')
    # os.system('vcp vos:' + maskLoc + ' ./ --verbose')

    cat = Table.read('SXDS_uddd_11.15_0.6.positions.fits')
    cat = cat[cat['tract'] == tract]
    cat = cat[cat['patch'] == patch]
    patch = im.split('_')[2].replace('.fits', '')
    print('Number of massive gal in this patch:', len(cat))

    # cut sources
    for gal in cat:
        print(gal['ID'])
        image = fits.open(im)
        w = wcs.WCS(image[0].header)
        y_shape, x_shape = image[0].shape[0], image[0].shape[1]
        ra_max, dec_min = w.wcs_pix2world(0, 0, 1)
        ra_min, dec_max = w.wcs_pix2world(x_shape, y_shape, 1)

        # cut for source and its mask
        x_gal, y_gal = w.wcs_world2pix(gal['ra'], gal['dec'], 0)
        if abs(x_gal-x_shape/2.) < x_shape/2.-225 and abs(y_gal-y_shape/2.) < y_shape/2.-225:
            cutoutimg(im, x_gal, y_gal, xw=225, yw=225, units='pixels',
                      outfile='cutout_'+gal['ID']+'.fits')
            cutoutimg(mask, x_gal, y_gal, xw=225, yw=225, units='pixels',
                      outfile='cutout_'+gal['ID']+'_mask.fits')

            for i in range(4):
                condition = False
                while not condition:
                    x_rand = 225 + random()*(image[0].shape[1]-225)
                    y_rand = 225 + random()*(image[0].shape[0]-225)
                    condition = abs(x_rand-x_shape/2.) < x_shape/2.-225 and abs(y_rand-y_shape/2.) < y_shape/2.-225
                    # cut for random and its mask

                cutoutimg(im, x_rand, y_rand, xw=225, yw=225, units='pixels',
                          outfile='cutout_'+gal['ID']+'_'+str(i)+'_rand.fits')
                cutoutimg(mask, x_rand, y_rand, xw=225, yw=225, units='pixels',
                          outfile='cutout_'+gal['ID']+'_'+str(i)+'_rand_mask.fits')

            gal_ims.write('cutout_'+gal['ID']+'.fits \n')

        else:
            print('Gal on the edge.')
