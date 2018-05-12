from uncertainties import ufloat
import numpy as np


def cal_error(n_sample, n_bkg, n_central):
    errors = []
    n_central = ufloat(n_central, np.sqrt(n_central))
    for i in range(len(n_sample)):
        n_sample_i = ufloat(n_sample[i], np.sqrt(n_sample[i]))
        n_bkg_i = ufloat(n_bkg[i], np.sqrt(n_bkg[i]))
        normed_n = (n_sample_i - n_bkg_i)/n_central
        errors.append(list(normed_n.error_components().items())[0][1])
    return np.array(errors)


def cal_error2(n_sample, n_bkg, n_central):
    n_comp = n_sample - n_bkg
    sigma_n_comp = np.sqrt(n_sample + n_bkg)
    sigma_n_central = np.sqrt(n_central)

    errors = (n_comp/n_central)*np.sqrt((sigma_n_comp/n_comp)**2+(sigma_n_central/n_central)**2)
    return errors
