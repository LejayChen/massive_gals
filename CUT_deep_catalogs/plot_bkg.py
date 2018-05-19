from astropy.table import Table
import matplotlib.pyplot as plt


def plot_bkg(cat_name, z):

    cat_data = Table.read('CUT_'+cat_name+'.fits')
    cat_gal = cat_data[cat_data['preds_median'] < 0.89]
    cat_massive_gal = cat_gal[cat_gal['MASS_MED'] > 11.15]
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal['zKDEPeak'] - z) < 0.09]

    cat_rp = Table.read('random_points_'+cat_name+'_'+str(z)+'.fits')

    plt.figure(figsize=(12, 10))
    plt.plot(cat_data['RA'], cat_data['DEC'], '.k', alpha=0.1, markersize=0.15)
    plt.plot(cat_rp['RA'], cat_rp['DEC'], 'bx', markersize=5.5)
    plt.plot(cat_massive_z_slice['RA'], cat_massive_z_slice['DEC'], 'or', markersize=3)
    plt.xlabel('R.A.', fontsize=16)
    plt.ylabel('DEC.', fontsize=16)
    plt.grid()
    plt.title(cat_name, fontsize=16)
    plt.savefig('bkg_plot_deep/'+cat_name+'_'+str(z)+'_bkgplot.png')

if __name__ == 'main':
    cat_name = 'COSMOS_deep'
    z = 0.6
    plot_bkg(cat_name, z)