import aplpy
from cutout import *
import numpy as np
from astropy.cosmology import WMAP9
import numpy as np
from astropy.table import Table
from astropy.io import fits
from sklearn.neighbors import KDTree


def find_patch(ra, dec, field='XMM-LSS'):
	cat_patches = Table.read('../tracts_patches/'+field+'_patches.fits')
	for patch in cat_patches:
		# print(patch[0])
		if ra < patch['corner0_ra']-0.09 and ra > patch['corner1_ra']+0.09 and dec < patch['corner2_dec']-0.09 and dec > patch['corner1_dec']+0.09:
			return str(patch['patch'])[0:-2], str(patch['patch'])[-2:]
	return 0, 0


cat = Table(fits.getdata('CUT3_XMM-LSS_deep.fits'))
cat = cat[cat['zKDEPeak'] < 1]
cat_gal = cat[np.logical_and(cat['preds_median'] < 0.89, cat['inside'] == True)]
cat_gal = cat_gal[cat_gal['MASS_MED'] > 9.0]
gal = cat_gal[cat_gal['NUMBER'] == 8524650006521][0]
dis = WMAP9.angular_diameter_distance(gal['zKDEPeak']).value

# satellites
cat_neighbors_z_slice = cat_gal[abs(cat_gal['zKDEPeak'] - gal['zKDEPeak']) < 1.5 * 0.044 * (1 + gal['zKDEPeak'])]
cat_neighbors = cat_neighbors_z_slice[abs(cat_neighbors_z_slice['RA'] - gal['RA']) < 0.7 / dis / np.pi * 180]
cat_neighbors = cat_neighbors[abs(cat_neighbors['DEC'] - gal['DEC']) < 0.7 / dis / np.pi * 180]
ind = KDTree(np.array(cat_neighbors['RA', 'DEC']).tolist()).query_radius([(gal['RA'], gal['DEC'])],0.7 / dis / np.pi * 180)
cat_neighbors = cat_neighbors[ind[0]]
cat_neighbors = cat_neighbors[cat_neighbors['NUMBER'] != gal['NUMBER']]

ra_list = np.array([])
dec_list = np.array([])
for sat in cat_neighbors:
	ra_list = np.append(ra_list, sat['RA'])
	dec_list = np.append(dec_list, sat['DEC'])

print(len(ra_list))
width = 0.9 / dis / np.pi * 180
cutoutimg('chi2_8524_65.fits', gal['RA'], gal['DEC'], xw=width, yw=width, units='wcs', outfile=str(gal['NUMBER'])+'.fits',
           overwrite=True, useMontage=False, coordsys='celestial', verbose=False, centerunits=None)
fig = aplpy.FITSFigure(str(gal['NUMBER'])+'.fits',hdu=0,auto_refresh=True)
fig.show_grayscale(stretch='log',vmin=0.1,invert=True)
fig.show_circles([gal['RA']], [gal['DEC']],radius=0.7 / dis / np.pi * 180, color='r',linewidth=1.5)
fig.show_circles([gal['RA']], [gal['DEC']],radius=0.02 / dis / np.pi * 180, color='y')
fig.show_circles(ra_list,dec_list,radius=0.02 / dis / np.pi * 180, color='b',linewidth=1.5)
fig.save('../figures/central.png', dpi=300)

# ==========================================
ra_bkg = 36.26097584242261
dec_bkg = -4.094378902670783

# satellites
cat_neighbors_z_slice = cat_gal[abs(cat_gal['zKDEPeak'] - gal['zKDEPeak']) < 1.5 * 0.044 * (1 + gal['zKDEPeak'])]
cat_neighbors = cat_neighbors_z_slice[abs(cat_neighbors_z_slice['RA'] - ra_bkg) < 0.7 / dis / np.pi * 180]
cat_neighbors = cat_neighbors[abs(cat_neighbors['DEC'] - dec_bkg) < 0.7 / dis / np.pi * 180]

ind = KDTree(np.array(cat_neighbors['RA', 'DEC']).tolist()).query_radius([(ra_bkg,dec_bkg)],0.7 / dis / np.pi * 180)
cat_neighbors = cat_neighbors[ind[0]]

ra_list_bkg = np.array([])
dec_list_bkg = np.array([])
for sat in cat_neighbors:
	ra_list_bkg = np.append(ra_list_bkg,sat['RA'])
	dec_list_bkg = np.append(dec_list_bkg,sat['DEC'])
print(len(ra_list_bkg))
cutoutimg('chi2_8766_12.fits', ra_bkg, dec_bkg, xw=width, yw=width, units='wcs', outfile='bkg.fits',
		  overwrite=True, useMontage=False, coordsys='celestial', verbose=False, centerunits=None)

fig = aplpy.FITSFigure('bkg.fits',hdu=0,auto_refresh=True)
fig.show_grayscale(stretch='log',vmin=0.1,invert=True)
fig.show_circles(ra_bkg,dec_bkg,radius=0.7 / dis / np.pi * 180,color='r',linewidth=1.5)
fig.show_circles(ra_list_bkg,dec_list_bkg,radius=0.02 / dis / np.pi * 180,color='b',linewidth=1.5)
fig.save('../figures/'+'bkg.png',dpi=300)
