from cutout import cutoutimg
from astropy.table import Table
import os

ims = open('ims.txt').readlines()

for im in ims:
    im = im.rstrip()
    mask = im.replace('chi2', 'chi2Mask')
    tract = im.split('_')[1]
    patch = im.split('_')[2].replace('.fits', '')
    patch = patch[0]+','+patch[1]

    maskLoc = "clauds/anneya/s16a_v2/s16a_udddv2.2/SExChi2_chi2Masks/" + mask
    imLoc = "clauds/anneya/s16a_v2/s16a_udddv2.2/SExChi2_chi2Ims/" + im
    os.system('vcp vos:' + imLoc + ' ./ --verbose ')
    os.system('vcp vos:' + maskLoc + ' ./ --verbose')

    # cut sources
    cat = Table.read()

    cat = cat[cat['TRACT'] == tract]
    cat = cat[cat['PATCH'] == patch]

    for gal in cat:
        cutoutimg()
        cutoutimg()

        ra_min, dec_min =
        ra_max, dec_max =
        ra_rand =
        dec_rand =

        cutoutimg()
        cutoutimg()