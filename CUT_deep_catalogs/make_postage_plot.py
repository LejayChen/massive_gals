import numpy as np
from random import random
from make_rgb_image import *
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# canfar username:...
# password:...

sfprob_bins = 4
fig, axs = plt.subplots(3, sfprob_bins, sharex=True, sharey=True, figsize=(16, 12))
for i, z in enumerate([0.4,0.6,0.8]):
    print('redshift bin:', z)

    cat_central = Table.read('massive_gal_positions/isolated_COSMOS_deep_11.15_'+str(round(z, 1))+'.positions.fits')
    for j, sfprob in enumerate(np.linspace(0,1,sfprob_bins+1)[:-1]):
        cat_central_cut = cat_central[np.logical_and(cat_central['sfProb'] > sfprob, cat_central['sfProb'] < sfprob+1.0/sfprob_bins)]
        success = False
        while not success:
            rand_id = int(random() * len(cat_central_cut))
            gal = cat_central_cut[rand_id]

            rgb_img, success = cut_i_band_img(gal['TRACT'], gal['PATCH'], gal['RA'], gal['DEC'],'I')
            if success:
                # img = mpimg.imread(rgb_img)
                img = fits.open(rgb_img)
                axs[i][j].imshow(np.log(abs(img[0].data)), cmap='gray')
                axs[i][j].get_xaxis().set_visible(False)
                axs[i][j].get_yaxis().set_visible(False)
                print('I-band image made:', rgb_img)

plt.tight_layout()
fig.subplots_adjust(hspace=0)
fig.subplots_adjust(wspace=0)
plt.savefig('../figures/test_postage_stamp.png')
