from cut_img import cutoutimg
from astropy.table import Table
from astropy.io import fits
from astropy import wcs
from random import random
from mkdir import *
import numpy as np
import os, sys

'''
make cutout images for each position

need patch images as input

'''

cat_name = sys.argv[1]  # catalog name (field name)
n_cutout = 7  # number of cutout images per patch
cutout_size=600 #pix

ims = open('Ims/ims_'+cat_name+'_new.txt').readlines()  # chi2 patch images
rand1_ids = open('Rand1_ids/rand1_ids_'+cat_name+'.txt', 'w')  # random position IDs
cutout_path = '/scratch-deleted-2021-mar-20/lejay/'  # directory to save cutout images at each random position

# random point catalog
cat_random = Table.read('/home/lejay/random_point_cat/'+cat_name+'_random_point.fits')
cat_random = cat_random[np.logical_and(cat_random['inside'] != 0, cat_random['MASK']==False)]

for im in ims:
    im = im.rstrip()
    splitter = '-'
    tract = im.split(splitter)[-3]
    patch = im.split(splitter)[-2] + ',' + im.split(splitter)[-1].replace('.fits', '')
    print(tract, patch)

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
        if len(cat_random_cut)<100:
            print('len(cat_random_cut)=='+str(len(cat_random_cut)))
            continue

    except IOError:
        print(im+' not found.')
        continue

    # prepare random1 positions (5 for each patch)
    for i in range(10):
        rand_id1 = np.random.randint(0, len(cat_random_cut))
        ra_rand1 = cat_random_cut[rand_id1]['RA']
        dec_rand1 = cat_random_cut[rand_id1]['DEC']

        # mkdir for storing cutout images
        cutout_patch_path = cutout_path + tract + '_' + patch[0] + patch[-1] + '_' + str(i) + '/'
        mkdir(cutout_patch_path)  # store cutouts for each galaxy in individual folders

        # cut sources
        x_rand1, y_rand1 = w.wcs_world2pix(ra_rand1, dec_rand1, 0)
        if abs(x_rand1-x_shape/2.) < x_shape/2.-cutout_size/2 and abs(y_rand1-y_shape/2.) < y_shape/2.-cutout_size/2:  # (not on edge)
            # cut for source
            cutoutimg(im, x_rand1, y_rand1, xw=cutout_size/2, yw=cutout_size/2, units='pixels',
                      outfile=cutout_patch_path+'cutout_'+tract+'_'+patch[0]+patch[-1]+'_'+str(i)+'.fits')

            # n_cutout random2 positioned image cutouts for each random1 cutout
            # for each random1 position, cut 10 random2 positions (for later stacking and estimate of recover rate)
            random_count = 0
            while random_count < n_cutout:
                id_rand2 = int(random() * len(cat_random_cut))
                ra_rand2 = cat_random_cut[id_rand2]['RA']
                dec_rand2 = cat_random_cut[id_rand2]['DEC']
                x_rand, y_rand = w.wcs_world2pix(ra_rand2, dec_rand2, 0)

                # cutout at random position within limit of patch image
                if abs(x_rand-x_shape/2.) < x_shape/2.-cutout_size/2 and abs(y_rand-y_shape/2.) < y_shape/2.-cutout_size/2:
                    random_count += 1
                    cutoutimg(im, x_rand, y_rand, xw=cutout_size/2, yw=cutout_size/2, units='pixels',
                              outfile=cutout_patch_path + 'cutout_' + tract+'_'+patch[0]+patch[-1]+'_'+str(i)+'_'+str(random_count-1)+'_rand.fits')

            if random_count >= n_cutout:
                rand1_ids.write(tract+'_'+patch[0]+patch[-1]+'_'+str(i)+'\n')
            else:
                continue

        else:
            print('Position ' + str(i) + ' on the edge.')
