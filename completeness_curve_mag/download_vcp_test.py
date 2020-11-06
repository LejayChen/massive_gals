import os, sys

ims = open('ims_COSMOS_deep.txt').readlines()
for im in ims:
    im = im.rstrip()
    splitter = '-'
    tract = im.split(splitter)[-3]
    patch = im.split(splitter)[-2] + ',' + im.split(splitter)[-1].replace('.fits', '')
    print(tract, patch)

    # download appropriate image
    imLoc = "clauds/picouet/Chi2Images/"+im
    imSaveLoc = '/home/lejay/projects/def-sawicki/lejay/'
    os.system('vcp vos:' + imLoc + ' '+imSaveLoc+' --verbose ')
    print('vcp vos:' + imLoc + ' '+imSaveLoc+' --verbose ')
