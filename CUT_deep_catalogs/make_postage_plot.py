import numpy as np
from random import random
from make_rgb_image import *
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(20, 12))
for i in range(3):
    z = 0.4 + i/10*2
    print('redshift bin:', z)

    cat_central = Table.read('massive_gal_positions/isolated_COSMOS_deep_11.15_'+str(round(z, 1))+'.positions.fits')
    for sfprob in [[0.0,0.5],[0.5,1.0]]:
        cat_central_cut = cat_central[np.logical_and(cat_central['sfProb'] > sfprob[0], cat_central['sfProb'] < sfprob[1])]
        success = False
        while not success:
            rand_id = int(random() * len(cat_central_cut))
            gal = cat_central_cut[rand_id]

            rgb_img, success = cut_rgb_image(gal['TRACT'], gal['PATCH'], gal['RA'], gal['DEC'])
            if success:
                img = mpimg.imread(rgb_img)
                axs[i][int(sfprob[0]/0.49999)].imshow(img, cmap='binary')
                print('rgb image made:', rgb_img)

plt.savefig('../figures/test_postage_stamp.png')
