from astropy.table import Table
from astropy.cosmology import Planck15
from astropy.coordinates import SkyCoord, match_coordinates_sky
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from random import random

cat = Table.read('CUT_CLAUDS_HSC_VISTA_Ks23.3_PHYSPARAM_TM.fits')
cat_gal = cat[cat['CLASS'] == 0]
cat_massive_gal = cat_gal[cat_gal['MASS_BEST'] > 11.3]


def bkg(mass_central, ra_central, cat_all_z_slice):

    '''
        return a background correction for each central galaxy by looking at black pointings without massive galaxies.

        mass_central: mass of the central galaxy
        ra_central: ra value for central galaxy. if ra_central>100, gal in COSMOS, if ra_central<100, gal in XMM-LSS
    '''

    counts_gals_rand = 0
    total_mass_sat_rand = 0
    total_mass_sat_sf_rand = 0
    total_mass_sat_q_rand = 0
    n = 0

    while n < 5:  # n is number of blank pointings

        same_field = False
        while same_field == False:
            id_rand = int(random()*len(cat_all_z_slice))
            ra_rand = cat_all_z_slice[id_rand]['RA']
            dec_rand = cat_all_z_slice[id_rand]['DEC']

            if ra_rand > 100 and ra_central>100:
                same_field = True
            elif ra_rand < 100 and ra_central<100:
                same_field = True
            else:
                same_field = False

        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(ra_rand, dec_rand, unit="deg"), coord_massive_gal, nthneighbor=1)

        if sep2d.degree > 1.0/dis/np.pi*180:  # make sure the blank pointing is away from any central galaxy
            cat_neighbors_rand = cat_all_z_slice[(cat_all_z_slice['RA'] - ra_rand) ** 2 + (cat_all_z_slice['DEC'] - dec_rand) ** 2 < (0.5 / dis / np.pi * 180) ** 2]

            # cut the satellite catalog
            mass_neighbors_rand = cat_neighbors_rand['MASS_BEST']
            mass_neighbors_sf_rand = cat_neighbors_rand[cat_neighbors_rand['SSFR_BEST'] > -11]['MASS_BEST']
            mass_neighbors_q_rand = cat_neighbors_rand[cat_neighbors_rand['SSFR_BEST'] > -11]['MASS_BEST']

            #  total mass for satellites in blank pointings (all, sf & q)
            total_mass_sat_rand += np.sum(10 ** (mass_neighbors_rand[mass_neighbors_rand>9] - 8))  # unit 10**8 M_sun
            total_mass_sat_sf_rand += np.sum(10 ** (mass_neighbors_sf_rand[mass_neighbors_sf_rand > 9] - 8))  # unit 10**8 M_sun
            total_mass_sat_q_rand += np.sum(10 ** (mass_neighbors_q_rand[mass_neighbors_q_rand > 9] - 8))  # unit 10**8 M_sun

            # satellite counts in mass bins in blank pointings
            count_gal_rand, edges_rand = np.histogram(10 ** (mass_neighbors_rand - mass_central), np.arange(0, 1.01, 0.1))
            counts_gals_rand += count_gal_rand

            n = n + 1

    return counts_gals_rand/5., total_mass_sat_rand/5., total_mass_sat_sf_rand/5., total_mass_sat_q_rand/5.


#################### MAIN FUNCTION ####################
total_mass_sat_log = open('total_mass_sat', 'w')  # record total mass of satellites for redshift evolution
for z in np.arange(0.3, 2.5, 0.1):
    print('============='+str(z)+'================')
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal['ZPHOT']-z)<0.1]  # massive galaxies in this z slice
    cat_all_z_slice = cat_gal[abs(cat_gal['ZPHOT'] - z) < 0.1]  # all galaxies in this z slice

    dis = Planck15.angular_diameter_distance(z).value
    search_r = 0.5/dis/np.pi*180/0.17  # HSC plate scale 0.17 arcsec/pix

    cat_massive_z_slice['RA'].unit = u.deg
    cat_massive_z_slice['DEC'].unit = u.deg
    coord_massive_gal = SkyCoord.guess_from_table(cat_massive_z_slice)

    # search for central galaxies (exclude massive companions)
    sep_angle = 0.0
    mass = cat_massive_z_slice[0]['MASS_BEST']
    id = cat_massive_z_slice[0]['ID']
    for gal in cat_massive_z_slice:
        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(gal['RA'], gal['DEC'], unit="deg"), coord_massive_gal, nthneighbor=2)
        if sep2d.degree < 0.5/dis/np.pi*180:
            if round(sep2d.degree[0], 5) == round(sep_angle, 5):
                if gal['MASS_BEST'] > mass:
                    cat_massive_z_slice = cat_massive_z_slice[cat_massive_z_slice['ID'] != id]
                else:
                    cat_massive_z_slice = cat_massive_z_slice[cat_massive_z_slice['ID'] != gal['ID']]

            sep_angle = sep2d.degree[0]
            mass = gal['MASS_BEST']
            id = gal['ID']

    # counting neighbors for each massive central galaxy
    counts_gals = np.zeros(10)
    total_mass_sat = 0
    total_mass_sat_sf = 0
    total_mass_sat_q = 0
    for gal in cat_massive_z_slice:
        cat_all_z_slice = cat_all_z_slice[cat_all_z_slice['ID'] != gal['ID']]
        cat_neighbors = cat_all_z_slice[(cat_all_z_slice['RA']-gal['RA'])**2+(cat_all_z_slice['DEC']-gal['DEC'])**2 < (0.5/dis/np.pi*180)**2]

        mass_neighbors = cat_neighbors['MASS_BEST']
        mass_neighbors_sf = cat_neighbors[cat_neighbors['SSFR_BEST']>-11]['MASS_BEST']
        mass_neighbors_q = cat_neighbors[cat_neighbors['SSFR_BEST']<-11]['MASS_BEST']
        mass_central = gal['MASS_BEST']

        # total mass of satellites (all, sf & q)
        total_mass_sat += np.sum(10**(mass_neighbors[mass_neighbors>9] - 8))  # unit 10**8 M_sun
        total_mass_sat_sf += np.sum(10**(mass_neighbors_sf[mass_neighbors_sf>9] - 8))  # unit 10**8 M_sun
        total_mass_sat_q += np.sum(10**(mass_neighbors_q[mass_neighbors_q>9] - 8))  # unit 10**8 M_sun

        # satellite counts in mass bins
        count_gal, edges = np.histogram(10**(mass_neighbors-mass_central), np.arange(0, 1.01, 0.1))
        count_gal = np.array(count_gal, dtype='float64')

        # background/foreground correction
        count_gal_bkg, mass_sat_bkg, mass_sat_sf_bkg, mass_sat_q_bkg = bkg(mass_central, gal['RA'], cat_all_z_slice)
        count_gal -= count_gal_bkg
        total_mass_sat -= mass_sat_bkg
        total_mass_sat_sf -= mass_sat_sf_bkg
        total_mass_sat_q -= mass_sat_q_bkg

        counts_gals += count_gal

    ######### PLOT ###################
    # plot sat counts against mass fraction M/M_central

    fig = plt.figure(figsize=(7, 6))
    plt.rc('font', family='serif'), plt.rc('xtick', labelsize=15), plt.rc('ytick', labelsize=15)

    plt.errorbar(x=np.arange(0, 0.91, 0.1)+0.05, y=counts_gals/len(cat_massive_z_slice), yerr=np.sqrt(counts_gals)/len(cat_massive_z_slice), fmt='.k', markersize=16, capsize=3, elinewidth=1)
    plt.ylabel('neighbor counts', fontsize=15)
    plt.xlabel(r'$M_{sat}/M_{central}$', fontsize=16)
    plt.annotate('z='+str(z-0.1)+'~'+str(z+0.1), (1, 11), fontsize=14, xycoords='axes points')
    plt.annotate(r'total mass of satellites: '+str(round(total_mass_sat/len(cat_massive_z_slice)))+'$*10^8 M_{\odot}$', (1,1), fontsize=14, xycoords='axes points')
    plt.annotate(r'total mass of sf satellites: ' + str(round(total_mass_sat_sf / len(cat_massive_z_slice))) + '$*10^8 M_{\odot}$', (1, 1), color='blue',fontsize=14, xycoords='axes points')
    plt.annotate(r'total mass of q satellites: ' + str(round(total_mass_sat_q / len(cat_massive_z_slice))) + '$*10^8 M_{\odot}$', (1, 1), color='red', fontsize=14, xycoords='axes points')

    plt.yscale('log')
    plt.ylim([0.001, 5])
    plt.savefig('companion_counts_plots/'+str(z)+'.png')
    plt.close()

    total_mass_sat_log.write(str(z) + ' ' + str(round(total_mass_sat / len(cat_massive_z_slice))) +
                             ' '+ str(round(total_mass_sat_sf / len(cat_massive_z_slice))) +
                             ' ' + str(round(total_mass_sat_q / len(cat_massive_z_slice))) +'\n')

total_mass_sat_log.close()