import numpy as np
import os, sys
from astropy.io import fits
from astropy import table, wcs
from mpi4py import MPI
from decide_cutout_number import *
from mkdir import *
comm=MPI.COMM_WORLD


def sextract(image, catalogName, flux_radii=[0.25, 0.5, 0.75], checkIm=None, checkImType=None):
    if checkIm==None:
        cmd = 'sex '+image+' -c chi2.sex -CATALOG_NAME '+ catalogName
    else:
        cmd = 'sex '+image+' -c chi2.sex -CATALOG_NAME '+ catalogName+' -CHECKIMAGE_TYPE '+checkImType+' -CHECKIMAGE_NAME '+checkIm+' -PSF_NAME default.psf'
    os.system(cmd)

    # Expand FLUX_RADIUS column
    try:
        t_sex = table.Table(fits.getdata(catalogName, 1))
        for col, rad in enumerate(flux_radii):
            t_sex['FLUX_RADIUS_'+str(rad)] = np.array(t_sex['FLUX_RADIUS'])[:, col]
        t_sex.remove_column('FLUX_RADIUS')
        t_sex.write(catalogName, overwrite=True)
        return 0
    except KeyError:
        return 1


def addImages(im1, im2, scale1, scale2, imOut):
    try:
        i1, i2 = fits.getdata(im1), fits.getdata(im2)
        i3 = scale1*i1 + scale2*i2
        fits.writeto(imOut, i3, header=fits.getheader(im1), overwrite=True)
        return 0
    except OSError:
        return 1


def renameColumns(cat, suffix):
    t = table.Table(fits.getdata(cat, 1))
    for col in t.colnames:
        t.rename_column(col, col+suffix)
    t.write(cat, format='fits', overwrite=True)

# MPI & other settings
rank = comm.Get_rank()
nProcs = comm.Get_size()
cat_name = sys.argv[1]
z = str(sys.argv[2])
cutout_path = '/home/lejay/scratch/'
tmp_path = sys.argv[3]+'/' # temporary storage of cutout model images and catalogs

# read in galaxy cutout img list
ids = open('Gal_ids/gal_ids_'+cat_name+'_'+z+'.txt').readlines()[:30]
n_cutout, cutout_size = decide_cutout_number(z)

nEach = len(ids)//nProcs
if rank == nProcs-1:
    myIds = ids[rank*nEach:]
else:
    myIds = ids[rank*nEach:rank*nEach+nEach]

ids = np.array(ids)
myIds = np.array(myIds)
print('THIS IS PROCESS'+' '+str(rank)+' ##################################################')
print('n_cutout', n_cutout,'cutout_size', cutout_size)
print('myIDs:', myIds)

for idd in myIds:
    idd = idd.rstrip()
    im = idd + '/' + 'cutout_' + idd + '.fits'  # name of the background frame
    print(cutout_path+im)

    random_im_cutout_idds = np.random.choice(myIds, 1, replace=False)
    try:
        # SExtract background frame chi2 image
        mkdir(tmp_path+idd+'/')
        flag_bkg = sextract(cutout_path+im, tmp_path+im.replace('.fits', '_cat.fits'), flux_radii=[0.25, 0.5, 0.75],
                                checkIm=tmp_path+im.replace('.fits', '_models.fits'), checkImType='models')

        # load background frame catalog
        t = table.Table(fits.getdata(tmp_path+im.replace('.fits', '_cat.fits'), 1))  # read
        to_length = len(t)
        if to_length == 0:
            continue

        # add column flagging whether objects came from background or source frame
        t['ORIGINAL'] = np.ones(len(t)).astype(bool)  # 1 == extracted from the background frame
        t['RA_deShift'] = t['X_WORLD']
        t['DEC_deShift'] = t['Y_WORLD']
        t.write(tmp_path+im.replace('.fits', '_cat.fits'), format='fits', overwrite=True)  # write

        cat_each_count = 0
        for idd_count, cutout_idd in enumerate(random_im_cutout_idds):
            for i in range(n_cutout):
                cutout_idd = cutout_idd.rstrip()
                rand_im_bkg_source_name = cutout_idd + '/' + 'cutout_' + cutout_idd + '.fits'
                rand_im = rand_im_bkg_source_name.replace('.fits', '_' + str(i) + '_rand.fits')

                # SExtract random cutout image (source frame)
                mkdir(tmp_path + cutout_idd + '/')
                flag_source = sextract(cutout_path+rand_im, tmp_path+rand_im.replace('.fits', '_cat.fits'), flux_radii=[0.25, 0.5, 0.75],
                                           checkIm=tmp_path+rand_im.replace('.fits', '_models.fits'), checkImType='models')

                # load source frame catalog
                t = table.Table(fits.getdata(tmp_path+rand_im.replace('.fits', '_cat.fits'), 1))
                tr_length = len(t)

                if flag_bkg==0 and flag_source==0 and tr_length>0:
                    # add ra/dec for each shifted object in source frame
                    t['ORIGINAL'] = np.zeros(len(t)).astype(bool)  # 0 == added from random shifted imgs
                    t['RA_deShift'] = t['X_WORLD']  # actual coord (deshifted)
                    t['DEC_deShift'] = t['Y_WORLD']   # actual coord (deshifted)
                    hdulist = fits.open(cutout_path+im)
                    w = wcs.WCS(hdulist[0].header)
                    pixCoords = np.c_[t['X_IMAGE'], t['Y_IMAGE']]
                    new_coord = w.wcs_pix2world(pixCoords, 1)  # source frame objects' world coords on source frame
                    t['X_WORLD'] = new_coord[:, 0]  # new coord
                    t['Y_WORLD'] = new_coord[:, 1]  # new coord
                    t.write(tmp_path+rand_im.replace('.fits', '_cat.fits'), format='fits', overwrite=True)

                    # join the source/background tables to make the base
                    to = table.Table(fits.getdata(tmp_path+im.replace('.fits', '_cat.fits')))  # bkg frame cat
                    tr = table.Table(fits.getdata(tmp_path+rand_im.replace('.fits', '_cat.fits')))  # source frame cat
                    tAll = table.join(to, tr, join_type='outer')
                    tAll.write(tmp_path+im.replace('.fits', '_'+str(i)+'_all_cat.fits'), format='fits', overwrite=True)

                    # add scaled & shifted chi2 images to original, run SExtractor on summed images
                    imSum = tmp_path+im.replace('.fits', '_chi2_sum.fits')
                    flag_add = addImages(cutout_path+im, tmp_path+rand_im.replace('.fits', '_models.fits'), 1.0, 1.0, imSum)
                    flag_sex_add = sextract(imSum, imSum.replace('.fits', '_cat.fits'))
                    renameColumns(imSum.replace('.fits', '_cat.fits'), '_1.0')

                    # merge/match new catalog with original+random
                    if flag_add == 0 and flag_sex_add == 0:
                        newCat = imSum.replace('.fits', '_cat.fits')  # objs from summed img
                        allCat = tmp_path+im.replace('.fits', '_'+str(i)+'_all_cat.fits')  # orignal objs + random objs
                        cmd = 'java -jar -Xms128m -Xmx256m stilts.jar tmatch2 progress=none in1='+allCat + \
                                ' in2='+newCat+' find=best join=all1 matcher=sky params=1 values1="X_WORLD Y_WORLD"' + \
                                ' values2="X_WORLD_1.0'+' Y_WORLD_1.0'+'" out='+allCat
                        os.system(cmd)

                        # stack the five catalogs from for random added cutout images (save in current directory)
                        cat_each = table.Table.read(allCat)
                        cat_each = cat_each['RA_deShift', 'DEC_deShift', 'X_WORLD', 'Y_WORLD', 'FLUX_APER', 'MAG_APER', 'FLUX_APER_1.0', 'ORIGINAL']

                        if idd_count==0 and cat_each_count==0:
                            cat_stack = cat_each
                        else:
                            cat_stack = table.vstack([cat_stack, cat_each], metadata_conflicts='silent')

                        cat_each_count += 1
                        print('merged: '+allCat)
                    else:
                        print('flag_add', flag_add, 'flag_sex_add', flag_sex_add)

                else:
                    print('to', to_length, 'tr', tr_length, 'flag_bkg', flag_bkg, 'flag_source', flag_source, cutout_idd, i)

        cat_stack.write(idd+'_all_cat.fits', overwrite=True)
        print('Written', idd + '_all_cat.fits')

    except FileNotFoundError as e:
        f = open('fails_' + cat_name + '.txt', 'a')
        print(im.replace(cutout_path, ''), file=f)
        print('File not found: ', e.filename)
        f.close()
