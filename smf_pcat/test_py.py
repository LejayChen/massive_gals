from astropy.table import Table

print('abc')

cat = Table.read('/home/lejay/projects/def-sawicki/lejay/phys_added_pcats/COSMOS_galaxies_241104.fits')
print(len(cat))
