from __future__ import print_function,division
import numpy as np
from astropy import units as u
from astropy.io import fits
import os
from astropy import table
import argparse
from astropy import wcs
from mpi4py import MPI
from rename import *
comm=MPI.COMM_WORLD
import tqdm

def sextract(image,catalogName,flux_radii=[0.25,0.5,0.75],checkIm=None,checkImType=None):
    if checkIm==None:
        cmd = 'sextractor '+image+' -c chi2.sex -CATALOG_NAME '+\
                catalogName
    else:
        cmd = 'sextractor '+image+' -c chi2.sex -CATALOG_NAME '+\
                catalogName+' -CHECKIMAGE_TYPE '+checkImType+' -CHECKIMAGE_NAME '+checkIm+' -PSF_NAME default.psf'
    print(cmd)
    os.system(cmd)

    # Expand FLUX_RADIUS
    t = table.Table(fits.getdata(catalogName, 1))
    for i, rad in enumerate(flux_radii):
        t['FLUX_RADIUS_'+str(rad)] = np.array(t['FLUX_RADIUS'])[:, i]
    t.remove_column('FLUX_RADIUS')
    t.write(catalogName, format='fits', overwrite=True)


def addMaskVal(mask,catalog,maskColName,xCol='X_IMAGE',yCol='Y_IMAGE'):
    m = fits.getdata(mask)
    cat = table.Table(fits.getdata(catalog))
    xRound = np.round(cat[xCol], 0).astype(int)
    yRound = np.round(cat[yCol], 0).astype(int)
    maskVals = m[yRound-1, xRound-1].astype(bool)
    cat[maskColName] = maskVals.astype(bool)
    cat.write(catalog, format='fits', overwrite=True)


def addImages(im1, im2, scale1, scale2, imOut):
    i1, i2 = fits.getdata(im1), fits.getdata(im2)
    i3 = scale1*i1 + scale2*i2
    fits.writeto(imOut, i3, header=fits.getheader(im1), overwrite=True)
    return imOut


def renameColumns(cat, suffix):
    t = table.Table(fits.getdata(cat, 1))
    for col in t.colnames:
        t.rename_column(col, col+suffix)
    t.write(cat, format='fits', overwrite=True)

#------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------------------------------------------------
# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--bands", nargs='+', help="bands to include in chi2 masks",
                    default=['MegaCam-uS', 'HSC-G','HSC-R','HSC-I','HSC-Z','HSC-Y'])
parser.add_argument("-p","--dataPath", help="full path to data release (where deepCoadds is a subfolder)",
                    default = './')
parser.add_argument("--tract",help='HSC tract id', default = '9813')
parser.add_argument("--overwrite", help='if True, will overwrite existing images',
                    default = False)
parser.add_argument("--prefix", help="prefix to be added to Chi2 images", default='uSgrizyChi2')
parser.add_argument("--dotsex", help="SExtractor .param file", default='default.sex')

args = parser.parse_args()
#------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------------------------------------------------
# MPI
rank = comm.Get_rank()
nProcs = comm.Get_size()
# ims = np.genfromtxt('ims.txt',dtype=None)
ims = open('gal_ims.txt').readlines()
nEach = len(ims)//nProcs
if rank == nProcs-1:
    myIms = ims[rank*nEach:]
else:
    myIms = ims[rank*nEach:rank*nEach+nEach]

if rank==0:
    myIms = tqdm.tqdm(myIms)
for im in myIms:
    if rank>=0:
        im = im.rstrip()
        rand_im = im.replace('.fits', '_rand.fits')
        mask = im.replace('.fits', '_mask.fits')
        rand_mask = rand_im.replace('.fits', '_mask.fits')

        print(im)
        print(rand_im)

        # SExtract orig
        # sextract(image,catalogName,flux_radii=[0.25,0.5,0.75],checkIm=None,checkImType=None)
        sextract(im, im.replace('.fits', '_cat.fits'), flux_radii=[0.25, 0.5, 0.75],
                 checkIm=im.replace('.fits', '_models.fits'), checkImType='models')

        # SExtract random
        sextract(rand_im, rand_im.replace('.fits', '_cat.fits'), flux_radii=[0.25, 0.5, 0.75],
                 checkIm=rand_im.replace('.fits', '_models.fits'), checkImType='models')

        # add column flagging whether objects came from original or random image
        t = table.Table(fits.getdata(im.replace('.fits', '_cat.fits'), 1))  # read
        t['ORIGINAL'] = np.ones(len(t)).astype(bool)
        t['RA_deShift'] = t['X_WORLD']
        t['DEC_deShift'] = t['Y_WORLD']
        t.write(im.replace('.fits', '_cat.fits'), format='fits', overwrite=True)  # write

        # add ra/dec of original object for each shifted object (so we can recover redshifts/physical properties)
        t = table.Table(fits.getdata(rand_im.replace('.fits', '_cat.fits'), 1))
        t['ORIGINAL'] = np.zeros(len(t)).astype(bool)
        t['RA_deShift'] = t['X_WORLD']
        t['DEC_deShift'] = t['Y_WORLD']
        hdulist = fits.open(im)
        w = wcs.WCS(hdulist[0].header)
        pixCoords = np.c_[t['X_IMAGE'], t['Y_IMAGE']]
        new_coord = w.wcs_pix2world(pixCoords, 1)  # random field objects' world coords on source frame
        t['X_WORLD'] = new_coord[:, 0]
        t['Y_WORLD'] = new_coord[:, 1]
        t.write(rand_im.replace('.fits', '_cat.fits'), format='fits', overwrite=True)

        # join the tables to make the base
        to = table.Table(fits.getdata(im.replace('.fits', '_cat.fits')))
        tr = table.Table(fits.getdata(rand_im.replace('.fits', '_cat.fits')))
        tAll = table.join(to, tr, join_type='outer')
        tAll.write(im.replace('.fits', '_all_cat.fits'), format='fits', overwrite=True)

        # add mask values from orig and rotated masks
        addMaskVal(mask, im.replace('.fits', '_all_cat.fits'), 'maskVal', xCol='X_IMAGE', yCol='Y_IMAGE')
        addMaskVal(rand_mask, im.replace('.fits', '_all_cat.fits'), 'shiftMaskVal', xCol='X_IMAGE', yCol='Y_IMAGE')

        # add scaled & rotated chi2 images to original, run SExtractor on summed images
        imSum = im.replace('.fits', '_chi2_sum.fits')
        addImages(im, rand_im.replace('.fits', '_models.fits'), 1.0, 1.0, imSum)
        sextract(imSum, imSum.replace('.fits', '_cat.fits'))
        renameColumns(imSum.replace('.fits', '_cat.fits'), '_1.0')

        # merge new catalog with original
        newCat = imSum.replace('.fits', '_cat.fits')
        allCat = im.replace('.fits', '_all_cat.fits')
        cmd = 'java -jar /home/lejay/.local/bin/stilts.jar tmatch2 in1='+allCat + \
            ' in2='+newCat+' find=best join=all1 matcher=sky params=1 values1="X_WORLD Y_WORLD"' + \
            ' values2="X_WORLD_1.0'+' Y_WORLD_1.0'+'" out='+allCat
        print(cmd)
        os.system(cmd)
        os.system('mv '+allCat+' ./'+allCat.replace('cutout', ''))
        os.system('rm cutout_*fits')
    else:
        f = open('./fails.txt', 'a')
        print(im, file=f)
        print('Failed!')
        f.close()