from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np


def completeness_est(mass_bin_edges, sfq):
    try:
        filename = '../mass_completeness_data/allFields_' + str(round(z - 0.1, 1)) + '_z_' + str(
            round(z + 0.1, 1)) + '_' + sfq + '_nopert_nan.txt'
        completeness_corr = np.genfromtxt(filename)

        completeness = np.array([])
        for i in range(len(mass_bin_edges)-1):
            mass = (mass_bin_edges[i]+mass_bin_edges[i+1])/2.
            completeness = np.append(completeness, np.interp(mass, completeness_corr[0], completeness_corr[3]))
        return completeness
    except IOError:
        print('no associated mass completeness file')
        return np.ones(len(mass_bin_edges)-1)

ssfr_cut = -10.5
bins = np.arange(7, 12, 0.05)
for z in np.arange(5, 5.1, 2)/10.:
    mass_list = np.array([])
    ssfr_list = np.array([])
    for cat_name in ['COSMOS_deep', 'SXDS_uddd', 'DEEP_deep', 'ELAIS_deep', 'XMM-LSS_deep']:
        print(cat_name)
        cat = Table.read('CUT3_' + cat_name + '.fits')
        cat = cat[cat['zKDEPeak'] < 1]
        cat_gal = cat[np.logical_and(cat['preds_median'] < 0.89, cat['inside'] == True)]
        cat_gal_z_slice = cat_gal[abs(cat_gal['zKDEPeak'] - z) < 0.1]
        mass_list = np.append(mass_list, cat_gal_z_slice['MASS_MED'])
        ssfr_list = np.append(ssfr_list, cat_gal_z_slice['SSFR_BEST'])

    mass_list = np.array(mass_list)
    ssfr_list = np.array(ssfr_list)

    print('plotting all...')
    all_hist, bin_edges = np.histogram(mass_list, bins=bins)
    all_hist = all_hist / completeness_est(bin_edges, 'all')
    all_hist[np.isnan(all_hist)] = 1

    print('plotting sf...')
    sf_mass_list = mass_list[ssfr_list > ssfr_cut]
    sf_hist, bin_edges = np.histogram(sf_mass_list, bins=bins)
    sf_hist = sf_hist / completeness_est(bin_edges, 'sf')
    sf_hist[np.isnan(sf_hist)] = 1

    print('plotting q...')
    q_mass_list = mass_list[ssfr_list < ssfr_cut]
    q_hist, bin_edges = np.histogram(q_mass_list, bins=bins)
    q_hist = q_hist / completeness_est(bin_edges, 'q')
    q_hist[np.isnan(q_hist)] = 1

    plt.hist(bin_edges[:-1]+0.025, weights=all_hist, bins=bins, histtype='step', color='k', label='z='+str(round(z,1))+',all')
    plt.hist(bin_edges[:-1]+0.025, weights=sf_hist, bins=bins, histtype='step', color='b', label='sf')
    plt.hist(bin_edges[:-1]+0.025, weights=q_hist, bins=bins, histtype='step', color='r', label='q')

    plt.yscale('log')
    plt.legend(fontsize=13)
    plt.savefig('../figures/ssfr_smf_all_fields'+str(round(z,1))+'.png')
    plt.show()