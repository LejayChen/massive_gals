from astropy.table import Table
from astropy.cosmology import WMAP9
from astropy.coordinates import SkyCoord, match_coordinates_sky
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from random import random

cat = Table.read('CUT_CLAUDS_HSC_VISTA_Ks23.3_PHYSPARAM_TM.fits')
cat_gal = cat[cat['CLASS'] == 0]
cat_massive_gal = cat_gal[cat_gal['MASS_BEST'] > 11.3]


def bkg(mass_cen, ra_central, cat_all_z_slice_rand, coord_massive_gal_rand):

    '''
        return a background correction for each central galaxy by looking at black pointings without massive galaxies.

        mass_central: mass of the central galaxy
        ra_central: ra value for central galaxy. if ra_central>100, gal in COSMOS, if ra_central<100, gal in XMM-LSS
    '''

    # __init__
    counts_gals_rand = 0
    mass_sat_rand = []
    mass_sat_sf_rand = []
    mass_sat_q_rand = []
    n = 0
    num_p = 8.0  # number of blank pointings

    # get several blank pointings to estimate background
    while n < int(num_p):

        # find a random pointing in the same field as the central galaxy
        same_field = False
        while not same_field:
            id_rand = int(random()*len(cat_all_z_slice_rand))
            ra_rand = cat_all_z_slice[id_rand]['RA']
            dec_rand = cat_all_z_slice[id_rand]['DEC']

            if ra_rand > 100 and ra_central > 100:
                same_field = True
            elif ra_rand < 100 and ra_central < 100:
                same_field = True
            else:
                same_field = False

        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(ra_rand, dec_rand, unit="deg"), coord_massive_gal_rand, nthneighbor=1)

        if sep2d.degree > 3.0/dis/np.pi*180:  # make sure the random pointing is away from any central galaxy (blank)

            # cut all objects catalog to get neighbor catalog
            coord_all_z_slice_rand = SkyCoord(cat_all_z_slice_rand['RA']*u.deg, cat_all_z_slice_rand['DEC']*u.deg)
            coord_rand = SkyCoord(ra_rand*u.deg, dec_rand*u.deg)
            cat_neighbors_rand = cat_all_z_slice_rand[coord_all_z_slice_rand.separation(coord_rand).degree < 0.5/dis/np.pi*180]

            # retrieve mass info in sat catalog (all, sf & q)
            mass_neighbors_rand = cat_neighbors_rand['MASS_BEST']
            mass_neighbors_sf_rand = cat_neighbors_rand[cat_neighbors_rand['SSFR_BEST'] > -11]['MASS_BEST']
            # mass_neighbors_q_rand = cat_neighbors_rand[cat_neighbors_rand['SSFR_BEST'] < -11]['MASS_BEST']

            # calculate total mass for satellites in blank pointings (all, sf & q)
            mass_sat_rand.append(np.sum(10 ** (mass_neighbors_rand[mass_neighbors_rand > 10] - 8)))  # unit 10**8 M_sun
            mass_sat_sf_rand.append(np.sum(10 ** (mass_neighbors_sf_rand[mass_neighbors_sf_rand > 10] - 8)))  # unit 10**8 M_sun
            # mass_sat_q_rand.append(np.sum(10 ** (mass_neighbors_q_rand[mass_neighbors_q_rand > 10] - 8)))   # unit 10**8 M_sun
            if len(mass_neighbors_rand[mass_neighbors_rand > 10]) == 0: continue

            # calculate satellite counts in mass bins in blank pointings
            count_gal_rand, edges_rand = np.histogram(10 ** (mass_neighbors_rand - mass_cen), np.arange(0, 1.01, 0.1))
            counts_gals_rand += count_gal_rand

            n = n + 1
    # stats
    total_mass_sat_rand = np.mean(mass_sat_rand)
    total_mass_sat_sf_rand = np.mean(mass_sat_sf_rand)
    total_mass_sat_q_rand = total_mass_sat_rand - total_mass_sat_sf_rand

    std_mass_sat_rand = np.std(mass_sat_rand)
    std_mass_sat_sf_rand = np.std(mass_sat_sf_rand)
    std_mass_sat_q_rand = np.std(np.array(mass_sat_rand) - np.array(mass_sat_sf_rand))

    return counts_gals_rand/num_p, total_mass_sat_rand, total_mass_sat_sf_rand, total_mass_sat_q_rand,\
           std_mass_sat_rand, std_mass_sat_sf_rand,  std_mass_sat_q_rand

# ################### MAIN FUNCTION ################### #

total_mass_sat_log = open('total_mass_sat', 'w')  # record total mass of satellites for redshift evolution
for z in np.arange(1.3, 1.4, 0.1):
    print('============='+str(z)+'================')
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal['ZPHOT'] - z) < 0.1]  # massive galaxies in this z slice
    cat_all_z_slice = cat_gal[abs(cat_gal['ZPHOT'] - z) < 0.1]  # all galaxies in this z slice
    dis = WMAP9.angular_diameter_distance(z).value  # angular diameter distance at redshift z

    # Fetch coordinates for massive gals
    cat_massive_z_slice['RA'].unit = u.deg
    cat_massive_z_slice['DEC'].unit = u.deg
    coord_massive_gal = SkyCoord.guess_from_table(cat_massive_z_slice)

    # counting neighbors and calculating total mass of neighbors for each massive central galaxy
    # __init__
    counts_gals = np.zeros(10)
    massive_counts = len(cat_massive_z_slice)
    mass_sat = []  # contains mass of sats
    mass_sat_sf = []
    mass_sat_q = []
    stds_mass_sat_bkg = []  # contains std errors of total mass in blank pointings
    stds_mass_sat_sf_bkg = []
    stds_mass_sat_q_bkg = []

    for gal in cat_massive_z_slice:

        # cut the z_slice catalog to get neighbors catalog
        coord_all_z_slice = SkyCoord(cat_all_z_slice['RA'] * u.deg, cat_all_z_slice['DEC'] * u.deg)
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)
        cat_neighbors = cat_all_z_slice[coord_all_z_slice.separation(coord_gal).degree < 0.5/dis/np.pi*180]
        cat_neighbors = cat_neighbors[cat_neighbors['ID'] != gal['ID']]

        if len(cat_neighbors) == 0:  # exlucde central gals which has no companion
            massive_counts -= 1
            continue

        # retrieve mass info in sat catalog (all, sf & q)
        mass_neighbors = cat_neighbors['MASS_BEST']
        if gal['MASS_BEST'] < max(cat_neighbors['MASS_BEST']):  # exclude central gals which has larger mass companion
            massive_counts -= 1
            continue
        mass_neighbors_sf = cat_neighbors[cat_neighbors['SSFR_BEST'] > -11]['MASS_BEST']
        # mass_neighbors_q = cat_neighbors[cat_neighbors['SSFR_BEST'] < -11]['MASS_BEST']
        mass_central = gal['MASS_BEST']

        # satellite counts in mass bins
        count_gal, edges = np.histogram(10**(mass_neighbors - mass_central), np.arange(0, 1.01, 0.1))
        count_gal_bkg, mass_sat_bkg, mass_sat_sf_bkg, mass_sat_q_bkg, std_mass_sat_bkg, std_mass_sat_sf_bkg, std_mass_sat_q_bkg = bkg(mass_central, gal['RA'], cat_all_z_slice, coord_massive_gal)
        counts_gals = counts_gals + (np.array(count_gal, dtype='float64') - count_gal_bkg)  # total sat_gal counts (binned with mass fraction)

        # total mass of satellites (all, sf & q; all=sf+q)
        # background substracted
        mass_sat.append(np.sum(10 ** (mass_neighbors[mass_neighbors > 10] - 8)) - mass_sat_bkg)  # unit 10**8 M_sun
        mass_sat_sf.append(np.sum(10 ** (mass_neighbors_sf[mass_neighbors_sf > 10] - 8)) - mass_sat_sf_bkg)  # unit 10**8 M_sun
        # mass_sat_q.append(np.sum(10 ** (mass_neighbors_q[mass_neighbors_q > 10] - 8)) - mass_sat_q_bkg)  # unit 10**8 M_sun
        stds_mass_sat_bkg.append(std_mass_sat_bkg)
        stds_mass_sat_sf_bkg.append(std_mass_sat_sf_bkg)
        stds_mass_sat_q_bkg.append(std_mass_sat_q_bkg)

    # averaged total mass of satellites

    total_mass_sat = np.mean(mass_sat)
    total_mass_sat_sf = np.mean(mass_sat_sf)
    total_mass_sat_q = total_mass_sat - total_mass_sat_sf

    # averaged error in total mass of satellites
    # m = m_cen - m_bkg
    # sigma_m = sqrt(sigma_cen**2 + sigma_bkg**2)

    std_mass_sat_total = np.sqrt(np.std(mass_sat)**2 + np.median(stds_mass_sat_bkg)**2)
    std_mass_sat_sf_total = np.sqrt(np.std(mass_sat_sf) ** 2 + np.median(stds_mass_sat_sf_bkg) ** 2)
    std_mass_sat_q_total = np.sqrt(np.std(np.array(mass_sat) - np.array(mass_sat_sf)) ** 2 + np.median(stds_mass_sat_q_bkg) ** 2)
    print(np.std(mass_sat), np.median(stds_mass_sat_bkg), std_mass_sat_total)

    # ######## PLOT #########
    # plot sat counts against mass fraction M/M_central

    fig = plt.figure(figsize=(7, 6))
    plt.rc('font', family='serif'), plt.rc('xtick', labelsize=15), plt.rc('ytick', labelsize=15)

    plt.errorbar(x=np.arange(0, 0.91, 0.1)+0.05, y=counts_gals/massive_counts, yerr=np.sqrt(counts_gals)/massive_counts, fmt='.k', markersize=16, capsize=3, elinewidth=1)
    plt.ylabel('neighbor counts', fontsize=16)
    plt.xlabel(r'$M_{sat}/M_{central}$', fontsize=16)
    plt.annotate('z='+str(z-0.1)+'~'+str(z+0.1)+', total:'+str(massive_counts), (1, 46), fontsize=14, xycoords='axes points')
    plt.annotate(r'total mass of satellites: '+str(round(total_mass_sat))+'$*10^8 M_{\odot}$', (1,31), fontsize=14, xycoords='axes points')
    plt.annotate(r'total mass of sf satellites: ' + str(round(total_mass_sat_sf)) + '$*10^8 M_{\odot}$', (1, 16), color='blue',fontsize=14, xycoords='axes points')
    plt.annotate(r'total mass of q satellites: ' + str(round(total_mass_sat_q)) + '$*10^8 M_{\odot}$', (1, 1), color='red', fontsize=14, xycoords='axes points')

    plt.yscale('log')
    plt.ylim([0.0005, 5])
    plt.savefig('companion_counts_plots/'+str(z)+'.png')
    plt.close()

    # output total sat masses to file
    total_mass_sat_log.write(str(z) + ' ' + str(round(total_mass_sat)) +
                             ' '+ str(round(total_mass_sat_sf)) +
                             ' ' + str(round(total_mass_sat_q)) +
                             ' '+str(round(std_mass_sat_total))+
                             ' '+str(round(std_mass_sat_sf_total))+
                             ' '+str(round(std_mass_sat_q_total))+
                             '\n')

total_mass_sat_log.close()