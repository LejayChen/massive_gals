import numpy as np

def cal_error2(n_sample, n_bkg, n_central):
    n_comp = np.array(n_sample - n_bkg).astype(float)
    n_comp[n_comp==0] = 1
    sigma_n_comp = np.array(np.sqrt(n_sample + n_bkg)).astype(float)
    sigma_n_central = np.array(np.sqrt(n_central)).astype(float)

    errors = (n_comp/n_central)*np.sqrt((sigma_n_comp/n_comp)**2+(sigma_n_central/n_central)**2)
    return errors


def cal_error3(n_sample, n_bkg, n_central, completeness, completeness_err):
    # number density = n_comp/n_central
    n_comp = np.array(n_sample - n_bkg).astype(float)
    n_comp[n_comp==0] = 1
    sigma_n_comp = np.array(np.sqrt(n_sample + n_bkg)).astype(float)
    sigma_n_comp = np.sqrt(sigma_n_comp**2*completeness**2 + n_comp**2*completeness_err**2)
    sigma_n_central = np.array(np.sqrt(n_central)).astype(float)

    errors = (n_comp/n_central)*np.sqrt((sigma_n_comp/n_comp)**2+(sigma_n_central/n_central)**2)
    return errors