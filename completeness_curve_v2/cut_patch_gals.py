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
print('python', cat_name, z)
n_cutout, cutout_size = decide_cutout_number(z)

# filenames and paths
ims = open('Ims/ims_'+cat_name+'.txt').readlines()  # chi2 patch images
gal_ids = open('Gal_ids/gal_ids_'+cat_name+'_'+z+'.txt', 'w')
cutout_path = '/home/lejay/scratch/'

# load random point catalog
cat_random = Table.read('/home/lejay/random_point_cat/'+cat_name+'_random_point.fits')
cat_random = cat_random[np.logical_and(cat_random['inside'] != 0, cat_random['MASK']==False)]

print(ims)
for im in ims:
    im = im.rstrip()
    splitter = '-'
    tract = im.split(splitter)[-3]
    patch = im.split(splitter)[-2] + ',' + im.split(splitter)[-1].replace('.fits', '')
    print(tract, patch)

    # load central galaxy catalog
    cat_massive = Table.read('/home/lejay/radial_dist_code/central_cat/isolated_'+cat_name+'_11.15_'+z+'_massive.positions.fits')
    cat_massive = cat_massive[cat_massive['TRACT'] == eval(tract)]
    cat_massive = cat_massive[cat_massive['PATCH'] == patch]
    print('Number of massive gal in this patch'+'('+str(tract)+' '+str(patch)+') is:', len(cat_massive))

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
        if len(cat_random_cut) < 100:
            print('cut_patch_gals: len(cat_random_cut)=='+str(len(cat_random_cut)))
            continue

    except IOError:
        print('cut_patch_gals: '+im+' not found.')
        continue

    # cut sources
    for gal in cat_massive:
        x_gal, y_gal = w.wcs_world2pix(gal['RA'], gal['DEC'], 0)
        if abs(x_gal-x_shape/2.) < x_shape/2. - cutout_size/2 and abs(y_gal-y_shape/2.) < y_shape/2.-cutout_size/2:  # (not on edge)
            idd = cat_name+'_'+str(gal['ID'])
            file_cutout_path = cutout_path + idd + '/'
            mkdir(file_cutout_path)  # store cutouts for each galaxy in individual folders

            # cut for source
            if os.path.exists(file_cutout_path):
                try:
                    cutoutimg(im, x_gal, y_gal, xw=cutout_size/2, yw=cutout_size/2, units='pixels', outfile=file_cutout_path + 'cutout_'+idd+'.fits')
                except (FileNotFoundError, ValueError) as e:
                    print('background image position failed for '+file_cutout_path + idd+'.fits')
                    continue

            else:
                print(file_cutout_path+' does not exist!')
                continue

            # n_cutout random positioned image cutouts for each gal cutout
            random_count = 0
            random_count_check = 0
            while random_count < n_cutout:
                id_rand = int(random() * len(cat_random_cut))
                ra_rand = cat_random_cut[id_rand]['RA']
                dec_rand = cat_random_cut[id_rand]['DEC']
                x_rand, y_rand = w.wcs_world2pix(ra_rand, dec_rand, 0)
                random_count_check += 1

                # cutout at random position
                if abs(x_rand-x_shape/2.) < x_shape/2.-cutout_size/2 and abs(y_rand-y_shape/2.) < y_shape/2.-cutout_size/2:
                    random_count += 1
                    try:
                        cutoutimg(im, x_rand, y_rand, xw=cutout_size/2, yw=cutout_size/2, units='pixels',
                              outfile=file_cutout_path + 'cutout_'+idd + '_' + str(random_count - 1) + '_rand.fits')
                    except (FileNotFoundError, ValueError) as e:
                        print('random source image position failed for' + file_cutout_path + idd + '_' + str(random_count - 1) + '.fits')
                        continue
                elif random_count_check > 100:
                    break

            if random_count >= n_cutout:
                gal_ids.write(idd+'\n')
            else:
                continue

        else:
            print('Gal on the edge.')
