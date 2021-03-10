from astropy.table import Table, vstack
import os
import sys

cat_name = sys.argv[1]
cat_stack_dir = '/home/lejay/projects/def-sawicki/lejay/completeness_output_mock_cats/rand_pos/'
cat_stack_name = 'cat_stack_'+cat_name+'.fits'
ids = open('Rand1_ids/rand1_ids_'+cat_name+'.txt').readlines()

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
cat_stack.write(cat_stack_dir+cat_stack_name, overwrite=True)
print('Merged '+str(len(ids))+' catalogs into '+cat_stack_dir+cat_stack_name)

for idd in ids:
    idd = idd.rstrip()
    os.system('rm '+idd+'_all_cat.fits')