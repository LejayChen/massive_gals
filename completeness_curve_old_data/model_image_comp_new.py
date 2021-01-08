import numpy as np
from astropy.io import fits
import os, sys
from astropy import table
from astropy import wcs
from mpi4py import MPI
comm=MPI.COMM_WORLD
import tqdm


def sextract(image, catalogName, flux_radii=[0.25, 0.5, 0.75], checkIm=None, checkImType=None):

    if data_type == 'olddata' or data_type == 'olddata_newcat':
        setting_file = 'chi2_old.sex'
    else:
        setting_file = 'chi2.sex'

    if checkIm==None:
        cmd = 'sex '+image+' -c '+setting_file+' -CATALOG_NAME '+\
                catalogName
    else:
        cmd = 'sex '+image+' -c '+setting_file+' -CATALOG_NAME '+\
                catalogName+' -CHECKIMAGE_TYPE '+checkImType+' -CHECKIMAGE_NAME '+checkIm+' -PSF_NAME default.psf'
    os.system(cmd)

    # Expand FLUX_RADIUS
    t = table.Table(fits.getdata(catalogName, 1))
    for i, rad in enumerate(flux_radii):
        t['FLUX_RADIUS_'+str(rad)] = np.array(t['FLUX_RADIUS'])[:, i]
    t.remove_column('FLUX_RADIUS')
    t.write(catalogName, format='fits', overwrite=True)


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

# MPI & other settings
rank = comm.Get_rank()
nProcs = comm.Get_size()
cat_name = sys.argv[1]  # field name
data_type = sys.argv[2]
cutout_path = '/home/lejay/scratch/'

# read in galaxy cutout img list
ids = open('Rand1_ids/rand1_ids_'+cat_name+'_'+data_type+'.txt').readlines()   # ID of rand1 positions
n_cutout = 10

nEach = len(ids)//nProcs
if rank == nProcs-1:
    myIds = ids[rank*nEach:]
else:
    myIds = ids[rank*nEach:rank*nEach+nEach]

print('THIS IS PROCESS'+' '+str(rank)+' ##################################################')
print('myIDs:', myIds)
if rank == 0:
    myIds = tqdm.tqdm(myIds)
for idd in myIds:
    idd = idd.rstrip()
    if rank >= 0:
        im = cutout_path + idd + data_type + '/' + 'cutout_'+idd+'.fits'
        print(im)

        try:
            # SExtract original chi2 image
            sextract(im, im.replace('.fits', '_cat.fits'), flux_radii=[0.25, 0.5, 0.75],
                     checkIm=im.replace('.fits', '_models.fits'), checkImType='models')

            for i in range(n_cutout):
                rand_im = im.replace('.fits', '_'+str(i)+'_rand.fits')

                # SExtract random cutout image
                sextract(rand_im, rand_im.replace('.fits', '_cat.fits'), flux_radii=[0.25, 0.5, 0.75],
                     checkIm=rand_im.replace('.fits', '_models.fits'), checkImType='models')

                # add column flagging for source image
                t = table.Table(fits.getdata(im.replace('.fits', '_cat.fits'), 1))  # read
                t['ORIGINAL'] = np.ones(len(t)).astype(bool)  # 1 == extracted from the original img
                t['RA_deShift'] = t['X_WORLD']  # X_WORLD == ra (actual coord)
                t['DEC_deShift'] = t['Y_WORLD']  # Y_WORLD == dec (actual coord)
                t.write(im.replace('.fits', '_cat.fits'), format='fits', overwrite=True)  # write

                # (random image) add original ra/dec of random object for each shifted object
                t = table.Table(fits.getdata(rand_im.replace('.fits', '_cat.fits'), 1))
                t['ORIGINAL'] = np.zeros(len(t)).astype(bool)  # 0 == added from random shifted imgs
                t['RA_deShift'] = t['X_WORLD']  # actual coord (deshifted)
                t['DEC_deShift'] = t['Y_WORLD']   # actual coord (deshifted)

                # random field objects' world coords on source frame
                hdulist = fits.open(im)  # im is the original image
                w = wcs.WCS(hdulist[0].header)
                pixCoords = np.c_[t['X_IMAGE'], t['Y_IMAGE']]
                new_coord = w.wcs_pix2world(pixCoords, 1)
                t['X_WORLD'] = new_coord[:, 0]  # source frame RA for mock object
                t['Y_WORLD'] = new_coord[:, 1]  # source frame DEC for mock object
                t.write(rand_im.replace('.fits', '_cat.fits'), format='fits', overwrite=True)


                to_name = im.replace('.fits', '_cat.fits')
                tr_name = rand_im.replace('.fits', '_cat.fits')
                to = table.Table(fits.getdata(to_name))  # table of objects from original (bkg) cutout
                tr = table.Table(fits.getdata(tr_name))  # table of objects from random cutout

                # remove objects from random cutout that are too close to objects in original (bkg) cutout
                distance_cut = 2 / 3600.  # degree
                matched_name = im.replace('.fits', '_'+str(i)+'_all_cat_matched_totr.fits')
                cmd = 'java -jar -Xms128m -Xmx256m stilts.jar tmatch2 progress=none in1=' + tr_name + \
                      ' in2=' + to_name + ' find=best join=all1 matcher=sky params=20 values1="X_WORLD Y_WORLD"' + \
                      ' values2="X_WORLD' + ' Y_WORLD' + '" out=' + matched_name
                os.system(cmd)
                tr_matched = table.Table.read(matched_name)
                tr = tr[tr_matched['Separation'] > distance_cut]

                # join the orignal/random tables to make the base
                tAll = table.join(to, tr, join_type='outer')
                tAll.write(im.replace('.fits', '_' + str(i) + '_all_cat.fits'), format='fits', overwrite=True)

                # add scaled & shifted chi2 images to original, run SExtractor on summed images
                imSum = im.replace('.fits', '_chi2_sum.fits')
                addImages(im, rand_im.replace('.fits', '_models.fits'), 1.0, 1.0, imSum)
                sextract(imSum, imSum.replace('.fits', '_cat.fits'))
                renameColumns(imSum.replace('.fits', '_cat.fits'), '_1.0')

                # merge/match new catalog with original+random
                newCat = imSum.replace('.fits', '_cat.fits')  # objs from summed img
                allCat = im.replace('.fits', '_'+str(i)+'_all_cat.fits')  # orignal objs + random objs (tAll)
                cmd = 'java -jar -Xms128m -Xmx256m stilts.jar tmatch2 progress=none in1='+allCat + \
                ' in2='+newCat+' find=best join=all1 matcher=sky params=1 values1="X_WORLD Y_WORLD"' + \
                ' values2="X_WORLD_1.0'+' Y_WORLD_1.0'+'" out='+allCat
                os.system(cmd)

                # add columns indicating original detected position of mock galaxies
                cat_each = table.Table.read(allCat)
                rand1_id = eval(idd.split('_')[-1])
                rand1_col = table.Column(data=np.ones(len(cat_each))*rand1_id, name='rand1')
                rand2_col = table.Column(data=np.ones(len(cat_each))*i, name='rand2')
                cat_each.add_columns([rand1_col, rand2_col])
                print(cat_each.keys())
                
                # stack the 10 catalogs from for random added cutout images (save in current directory)
                if i == 0:
                    cat_stack = cat_each
                else:
                    cat_stack = table.vstack([cat_stack, cat_each], metadata_conflicts='silent')
                    print('merge with'+allCat)

            # os.system('rm -r '+'/home/lejay/scratch/'+idd)  # remove all cutuot images and temporary catalogs
            cat_stack.write(idd+'_all_cat.fits', overwrite=True)
            print('Written', idd+'_all_cat.fits')

        except FileNotFoundError:
            # os.system('rm -r '+'/home/lejay/scratch/'+idd)
            f = open('fails_'+cat_name+'.txt', 'a')
            print(im.replace(cutout_path, ''), file=f)
            print('Failed! '+im.replace(cutout_path, ''))
            f.close()
