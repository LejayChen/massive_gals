from astropy.table import Table, vstack
import os

ims = open('gal_ims.txt').readlines()
im0 = ims[0].rstrip()
cat_stack = Table.read(im0.replace('.fits', '_all_cat.fits').replace('cutout_',''))
cat_stack_name = 'cat_stack.fits'

for im in ims[1:]:
    im = im.rstrip()
    cat = Table.read(im.replace('.fits', '_all_cat.fits').replace('cutout_', ''))
    cat_stack = vstack([cat_stack, cat])

cat_stack.write(cat_stack_name, overwrite=True)
os.system('rm *_all_cat.fits')

# match with phys catalog
cat_phys = '../CUT_deep_catalogs/CUT2_SXDS_uddd.fits'
cmd = 'java -jar /home/lejay/.local/bin/stilts.jar tmatch2 in1='+cat_stack_name + \
            ' in2='+cat_phys+' find=best join=all1 matcher=sky params=1 values1="RA_deShift DEC_deShift"' + \
            ' values2="RA'+' DEC'+'" out='+'matched_'+cat_stack_name
print(cmd)
os.system(cmd)