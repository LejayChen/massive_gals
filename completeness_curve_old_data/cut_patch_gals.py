from cut_img import cutoutimg
from astropy.table import Table
from astropy.io import fits
from astropy import wcs
from mkdir import *
import numpy as np
import os, sys

'''
make cutout images for each position
need patch images as input
'''

cat_name = sys.argv[1]  # catalog name (field name)
data_type = sys.argv[2]
n_cutout = 10  # number of cutout images per patch
cutout_size = 600  # pix

ims = open('Ims/ims_'+cat_name+'_10patches.txt').readlines()  # chi2 patch images
rand1_ids = open('Rand1_ids/rand1_ids_'+cat_name+'_'+data_type+'.txt', 'w')  # random position IDs
cutout_path = '/home/lejay/scratch/'  # directory to save cutout images at each random position

# data catalog
cat_gal = Table.read('/home/lejay/catalogs/v2_cats/DEEP_deep_gal_cut_params.fits')  # inside, unmasked, gals

# random point catalog
if data_type == 'newdata' or data_type == 'newdata_oldcat':
    cat_random = Table.read('/home/lejay/random_point_cat/'+cat_name+'_random_point.fits')
    cat_random = cat_random[np.logical_and(cat_random['inside'] != 0, cat_random['MASK']==False)]
else:
    cat_random = Table.read('/home/lejay/random_point_cat_old/s16a_'+cat_name+'_random.fits')
    if cat_name != 'XMM-LSS_deep':
        cat_random = cat_random[np.logical_and(cat_random['inside_u'] != 0, cat_random['MASK']==False)]
    else:
        cat_random = cat_random[np.logical_and(cat_random['inside_u'] != 0, cat_random['MASK'] == False)]

for im in ims:  # image of patches
    im = im.rstrip()
    splitter = '-'
    tract = im.split(splitter)[-3]
    patch = im.split(splitter)[-2] + ',' + im.split(splitter)[-1].replace('.fits', '')

    print(tract, patch)
    rand_pos = np.load('rand_pos_choice/'+tract+'_'+im.split(splitter)[-2]+im.split(splitter)[-1].replace('.fits', '')+'rand_pos.npy')

    # append path for im
    if data_type == 'newdata' or data_type == 'newdata_oldcat':
        imSaveLoc = '/home/lejay/projects/def-sawicki/lejay/new_chi2_imgs/'
        im = imSaveLoc + im
    else:
        imSaveLoc = '/home/lejay/scratch/old_chi2_imgs/'
        im = imSaveLoc + 'chi2_'+tract+'_'+patch[0]+patch[-1]+'.fits'

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

        # discard any patch that contains less than 100 random points
        if len(cat_random_cut) < 100:
            print('len(cat_random_cut)=='+str(len(cat_random_cut)))
            continue

    except IOError:
        print(im+' not found.')
        continue

    # prepare random1 positions (10 for each patch)
    for i in range(10):
        ra_rand1 = rand_pos[i][0][0]
        dec_rand1 = rand_pos[i][0][1]

        # no bright objects in rand1 cutout image
        cutout_size_degree = cutout_size * 4.666e-5
        cat_gal_cutout = cat_gal[abs(cat_gal['RA'] - ra_rand1)<cutout_size_degree/2]
        cat_gal_cutout = cat_gal_cutout[abs(cat_gal_cutout['DEC'] - dec_rand1) < cutout_size_degree/2]
        if max(cat_gal_cutout['i']) < 18:
            continue

        # cut sources
        x_rand1, y_rand1 = w.wcs_world2pix(ra_rand1, dec_rand1, 0)
        if abs(x_rand1-x_shape/2.) < x_shape/2.-cutout_size/2 and abs(y_rand1-y_shape/2.) < y_shape/2.-cutout_size/2:  # (not on edge of cutput image)
            cutout_additional_path = tract+'_'+patch[0]+patch[-1]+'_'+str(i)+data_type
            mkdir('/home/lejay/scratch/' + cutout_additional_path)  # store cutouts for each galaxy in individual folders

            # cut for source
            cutout_path = '/home/lejay/scratch/' + cutout_additional_path + '/'
            cutoutimg(im, x_rand1, y_rand1, xw=cutout_size/2, yw=cutout_size/2, units='pixels',
                      outfile=cutout_path+'cutout_'+tract+'_'+patch[0]+patch[-1]+'_'+str(i)+'.fits')

            # n_cutout random2 positioned image cutouts for each random1 cutout
            # for each random1 position, cut 10 random2 positions (for later stacking and estimate of recover rate)
            random_count = 0
            while random_count < n_cutout:
                ra_rand2 = rand_pos[i][random_count + 1][0]
                dec_rand2 = rand_pos[i][random_count + 1][1]
                x_rand2, y_rand2 = w.wcs_world2pix(ra_rand2, dec_rand2, 0)

                # cutout at random position
                if abs(x_rand2-x_shape/2.) < x_shape/2.-cutout_size/2 and abs(y_rand2-y_shape/2.) < y_shape/2.-cutout_size/2.:
                    random_count += 1
                    cutoutimg(im, x_rand2, y_rand2, xw=cutout_size/2, yw=cutout_size/2, units='pixels',
                              outfile=cutout_path + 'cutout_' + tract+'_'+patch[0]+patch[-1]+'_'+
                                      str(i) + '_' + str(random_count - 1) + '_rand.fits')
                else:
                    print(x_rand2, x_shape, y_rand2, y_shape)
                    print('Position rand2 ' + str(random_count) + ' on the edge.')
                    break

            if random_count >= n_cutout:
                rand1_ids.write(tract+'_'+patch[0]+patch[-1]+'_'+str(i)+'\n')
            else:
                continue

        else:
            print('Position rand1 ' + str(i) + ' on the edge.')
