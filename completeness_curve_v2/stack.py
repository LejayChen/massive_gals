from astropy.table import Table, vstack
import os
import sys

cat_name = sys.argv[1]
z = str(sys.argv[2])
output_path = 'Output_cats/'
cat_stack_name = 'cat_stack_'+cat_name+'_'+z+'.fits'
ids = open('Gal_ids/gal_ids_'+cat_name+'_'+z+'.txt').readlines()
cutout_path = '/scratch-deleted-2021-mar-20/lejay/'

cat_list = []
for idd in ids:
    im = 'cutout_'+idd.rstrip()+'.fits'
    try:
        cat = Table.read(im.replace('.fits', '_all_cat.fits').replace('cutout_', ''))
        cat = cat[cat['ORIGINAL'] == False]
        print('Read '+im.replace('.fits', '_all_cat.fits').replace('cutout_', ''))
        cat_list.append(cat)
    except FileNotFoundError:
        print(im.replace('.fits', '_all_cat.fits').replace('cutout_', '')+' not found!')

cat_stack = vstack(cat_list, metadata_conflicts='silent')
cat_stack.write(output_path + cat_stack_name, overwrite=True)
print('Merged '+str(len(ids))+' catalogs into '+cat_stack_name)

for idd in ids:
    idd = idd.rstrip()
    os.system('rm *_all_cat.fits')
