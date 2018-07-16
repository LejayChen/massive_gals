from __future__ import print_function,division
import numpy as np
from astropy import units as u
from astropy.io import fits
import os
from astropy import table
import argparse
from astropy import wcs
from mpi4py import MPI
comm=MPI.COMM_WORLD
import tqdm


def rotateChi2(image):
    im = fits.getdata(image)
    k = 1
    if im.shape[0]!=im.shape[1]:
        k = 2

    fits.writeto(image.replace('.fits','_rot.fits'),np.rot90(im,k=k),\
                     header=fits.getheader(image),overwrite=True)
    return k, im.shape


def rotXY(x,y,shape,k):
    for i in range(k):
        xShift = x-shape[1]/2
        yShift = y-shape[0]/2
        xRot, yRot = -1*yShift, xShift
        x, y = xRot+shape[0]/2, yRot+shape[1]/2
        shape = [shape[1], shape[0]]
    return x, y


def sextract(image,catalogName,flux_radii=[0.25,0.5,0.75],checkIm=None,checkImType=None):
    if checkIm==None:
        cmd = 'sextractor '+image+' -c chi2.sex -CATALOG_NAME '+\
                catalogName
    else:
        cmd = 'sextractor '+image+' -c chi2.sex -CATALOG_NAME '+\
                catalogName+' -CHECKIMAGE_TYPE '+checkImType+' -CHECKIMAGE_NAME '+checkIm+' -PSF_NAME default.psf'
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
    maskVals = m[yRound-1,xRound-1].astype(bool)
    cat[maskColName] = maskVals.astype(bool)
    cat.write(catalog, format='fits', overwrite=True)


def addImages(im1,im2,scale1,scale2,imOut):
    i1,i2 = fits.getdata(im1),fits.getdata(im2)
    i3 = scale1*i1 + scale2*i2
    fits.writeto(imOut,i3,header=fits.getheader(im1),overwrite=True)
    return(imOut)


def renameColumns(cat,suffix):
    t = table.Table(fits.getdata(cat,1))
    for col in t.colnames:
        t.rename_column(col, col+suffix)
    t.write(cat,format='fits',overwrite=True)

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
ims = open('ims.txt').readlines()
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
        mask = im.replace('chi2', 'chi2Mask')
        tract = im.split('_')[1]
        patch = im.split('_')[2].replace('.fits', '')

        maskLoc = "clauds/anneya/s16a_v2/s16a_udddv2.2/SExChi2_chi2Masks/"+mask
        imLoc = "clauds/anneya/s16a_v2/s16a_udddv2.2/SExChi2_chi2Ims/"+im
        os.system('vcp vos:'+imLoc+' ./ --verbose ')
        os.system('vcp vos:'+maskLoc+' ./ --verbose')

        im = './'+im
        mask = './'+mask

        # rotate
        k, shape = rotateChi2(im)
        rotateChi2(mask)

        # SExtract orig
        # sextract(image,catalogName,flux_radii=[0.25,0.5,0.75],checkIm=None,checkImType=None)
        sextract(im, im.replace('.fits', '_cat.fits'), flux_radii=[0.25, 0.5, 0.75],
                 checkIm=im.replace('.fits', '_models.fits'), checkImType='models')

        rotateChi2(im.replace('.fits', '_models.fits'))
        sextract(im.replace('.fits', '_rot.fits'), im.replace('.fits','_rot_cat.fits'))

        # add column flagging whether objects came from original or rotated image
        t = table.Table(fits.getdata(im.replace('.fits', '_cat.fits'), 1))
        print(t.info())
        t['ORIGINAL'] = np.ones(len(t)).astype(bool)
        t['RA_deRot'] = t['X_WORLD']
        t['DEC_deRot'] = t['Y_WORLD']
        t.write(im.replace('.fits', '_cat.fits'), format='fits', overwrite=True)

        # add ra/dec of original object for each rotated object (so we can recover redshifts/physical properties)
        t = table.Table(fits.getdata(im.replace('.fits', '_rot_cat.fits'), 1))
        t['ORIGINAL'] = np.zeros(len(t)).astype(bool)
        x = t['X_IMAGE']
        y = t['Y_IMAGE']
        xRot, yRot = rotXY(x, y, shape, k)
        hdulist = fits.open(im)
        w = wcs.WCS(hdulist[0].header)
        pixCoords = np.c_[xRot, yRot]
        derot=w.wcs_pix2world(pixCoords, 1)
        t['RA_deRot'] = derot[:, 0]
        t['DEC_deRot'] = derot[:, 1]
        t.write(im.replace('.fits', '_rot_cat.fits'), format='fits', overwrite=True)

        # join the tables to make the base
        to = table.Table(fits.getdata(im.replace('.fits', '_cat.fits')))
        tr = table.Table(fits.getdata(im.replace('.fits', '_rot_cat.fits')))
        tAll = table.join(to, tr, join_type='outer')
        tAll.write(im.replace('.fits', '_all_cat.fits'), format='fits', overwrite=True)

        # add mask values from orig and rotated masks
        addMaskVal(mask, im.replace('.fits', '_all_cat.fits'), 'maskVal', xCol='X_IMAGE', yCol='Y_IMAGE')
        addMaskVal(mask.replace('.fits', '_rot.fits'), im.replace('.fits', '_all_cat.fits'), 'rotMaskVal', xCol='X_IMAGE', yCol='Y_IMAGE')

        # add scaled & rotated chi2 images to original, run SExtractor on summed images
        fScales = np.arange(0.1, 1.1, 0.1)
        for fScale in fScales:
            print(fScale)
            newIm = im.replace('.fits', '_'+str(fScale)+'chi2_rot.fits')
            imSum = addImages(im, im.replace('.fits', '_models_rot.fits'), 1.0, fScale, newIm)
            sextract(newIm, newIm.replace('.fits', '_cat.fits'))
            renameColumns(newIm.replace('.fits', '_cat.fits'), '_'+str(fScale))

            # merge new catalog with original
            newCat = newIm.replace('.fits', '_cat.fits')
            allCat = im.replace('.fits', '_all_cat.fits')

            cmd = 'java -jar /home/lejay/.local/bin/stilts.jar tmatch2 in1='+allCat + \
            ' in2='+newCat+' find=best join=all1 matcher=sky params=1 values1="X_WORLD Y_WORLD"' + \
            ' values2="X_WORLD'+'_'+str(fScale)+' Y_WORLD'+'_'+str(fScale)+'" out='+allCat
            os.system(cmd)

        # os.system('rm ./*'+tract+'_'+patch+'*')
    else:
        f = open('./fails.txt', 'a')
        print(im, file=f)
        print('Failed!')
        f.close()













