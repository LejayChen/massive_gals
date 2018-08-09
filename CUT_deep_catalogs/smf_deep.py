from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np


def completeness_est(mass_list, ssfr_list, z):
    try:
        completeness_sf = np.genfromtxt('../mass_completeness_data/allFields_' + str(round(z - 0.1, 1)) + '_z_' + str(round(z + 0.1,1)) + '_sf_nopert_nan.txt')
        completeness_q = np.genfromtxt('../mass_completeness_data/allFields_' + str(round(z - 0.1, 1)) + '_z_' + str(round(z + 0.1,1)) + '_q_nopert_nan.txt')
        completeness = np.array([])
        for idx in range(len(mass_list)):
            if ssfr_list[idx] > -11:
                completeness = np.append(completeness, np.interp(mass_list[idx], completeness_sf[0], completeness_sf[3]))
            else:
                completeness = np.append(completeness, np.interp(mass_list[idx], completeness_q[0], completeness_q[3]))

        completeness[np.isnan(completeness)] = 1.
        return completeness
    except:
        return np.ones(len(mass_list))

for z in np.arange(3, 5.1, 2)/10.:
    mass_list = np.array([])
    ssfr_list = np.array([])
    sfprob_list = np.array([])
    for cat_name in ['COSMOS_deep', 'SXDS_uddd', 'DEEP_deep', 'ELAIS_deep', 'XMM-LSS_deep']:
        print(cat_name)
        cat = Table.read('CUT3_' + cat_name + '.fits')
        cat = cat[cat['zKDEPeak'] < 1]
        cat_gal = cat[np.logical_and(cat['preds_median'] < 0.89, cat['inside'] == True)]
        cat_gal_z_slice = cat_gal[abs(cat_gal['zKDEPeak'] - z) < 0.1]
        mass_list = np.append(mass_list, cat_gal_z_slice['MASS_MED'])
        ssfr_list = np.append(ssfr_list, cat_gal_z_slice['SSFR_BEST'])
        sfprob_list = np.append(sfprob_list,cat_gal_z_slice['sfProb'])

    print('plotting...')
    mass_list = np.array(mass_list)
    ssfr_list = np.array(ssfr_list)
    plt.hist(mass_list, weights=sfprob_list / completeness_est(mass_list, ssfr_list, z),
             bins=np.arange(7, 12, 0.1), histtype='step', label=str(z),color='k')

    plt.hist(mass_list, weights=sfprob_list / completeness_est(mass_list, ssfr_list, z),
             bins=np.arange(7, 12, 0.1), histtype='step', label=str(z), color='b')

    plt.hist(mass_list, weights=(1- sfprob_list) / completeness_est(mass_list, ssfr_list, z),
             bins=np.arange(7, 12, 0.1), histtype='step', label=str(z), color='r')

    plt.yscale('log')
    plt.legend()
    plt.savefig('../figures/sfprob_smf_all_fields'+str(z)+'.png')
    # plt.show()


