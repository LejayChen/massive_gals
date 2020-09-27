from cut_img import cutoutimg
from astropy.table import Table
from astropy.io import fits
from astropy import wcs
from random import random
from decide_cutout_number import *
from mkdir import *
import numpy as np
import os, sys

cat_name = sys.argv[1]
z = str(sys.argv[2])
ims = open('Ims/ims_'+cat_name+'.txt').readlines()
gal_ids = open('Gal_ids/gal_ids_'+cat_name+'_'+z+'.txt', 'w')
cutout_path = '/home/lejay/scratch/'
n_cutout = decide_cutout_number(z)
cat_random = Table.read('/home/lejay/random_point_cat/'+cat_name+'_random_point.fits')
cat_random = cat_random[np.logical_and(cat_random['inside'] != 0, cat_random['MASK']==False)]

for im in ims:
    im = im.rstrip()
    splitter = '-'
    tract = im.split(splitter)[-3]
    patch = im.split(splitter)[-2] + ',' + im.split(splitter)[-1].replace('.fits', '')
    print(tract, patch)

    # load central galaxy catalog
    cat = Table.read('/home/lejay/v2_matched_centrals/central_'+cat_name+'_'+z+'.fits')
    cat = cat[cat['TRACT'] == eval(tract)]
    cat = cat[cat['PATCH'] == patch]
    print('Number of massive gal in this patch:'+'('+str(tract)+' '+str(patch)+') is', len(cat))

    # append path for im
    imSaveLoc = '/home/lejay/projects/def-sawicki/lejay/new_chi2_imgs/'
    im = imSaveLoc + im

    try:
        # load chi2 images
        image = fits.open(im)
        w = wcs.WCS(image[0].header)

        # get shape
        y_shape, x_shape = image[0].shape[0], image[0].shape[1]
        ra_max, dec_min = w.wcs_pix2world(0, 0, 1)
        ra_min, dec_max = w.wcs_pix2world(x_shape, y_shape, 1)

        # cut random point catalog (within patch limit)
        cat_random_cut = cat_random[np.logical_and(cat_random['RA']>ra_min, cat_random['RA']<ra_max)]
        cat_random_cut = cat_random_cut[np.logical_and(cat_random_cut['DEC']>dec_min, cat_random_cut['DEC']<dec_max)]
        if len(cat_random_cut)==0:
            print('len(cat_random_cut)=='+str(len(cat_random_cut)))
            continue

    except IOError:
        print(im+' not found.')
        continue

    # cut sources
    for gal in cat:
        x_gal, y_gal = w.wcs_world2pix(gal['RA'], gal['DEC'], 0)
        if abs(x_gal-x_shape/2.) < x_shape/2.-225 and abs(y_gal-y_shape/2.) < y_shape/2.-225:  # (not on edge)
            mkdir('/home/lejay/scratch/'+str(gal['ID']))  # store cutouts for each galaxy in individual folders

            # cut for source
            cutout_path = '/home/lejay/scratch/'+str(gal['ID'])+'/'
            cutoutimg(im, x_gal, y_gal, xw=225, yw=225, units='pixels',
                      outfile=cutout_path+'cutout_'+gal['ID']+'.fits')

            # n_cutout random positioned image cutouts for each gal cutout
            random_count = 0
            while random_count < n_cutout:
                id_rand = int(random() * len(cat_random_cut))
                ra_rand = cat_random_cut[id_rand]['RA']
                dec_rand = cat_random_cut[id_rand]['DEC']
                x_rand, y_rand = w.wcs_world2pix(ra_rand, dec_rand, 0)

                # cutout at random position
                if abs(x_rand-x_shape/2.) < x_shape/2.-225 and abs(y_rand-y_shape/2.) < y_shape/2.-225:
                    random_count += 1
                    cutoutimg(im, x_rand, y_rand, xw=225, yw=225, units='pixels',
                              outfile=cutout_path + 'cutout_' + gal['ID'] + '_' + str(random_count - 1) + '_rand.fits')

            if random_count >= n_cutout:
                gal_ids.write(str(gal['ID'])+'\n')
            else:
                continue

        else:
            print('Gal on the edge.')
