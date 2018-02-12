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

def bkg(mass_cen, ra_central, cat_allobj_z_slice):

    '''
        return a background correction for each central galaxy by looking at black pointings without massive galaxies.

        mass_central: mass of the central galaxy
        ra_central: ra value for central galaxy. if ra_central>100, gal in COSMOS, if ra_central<100, gal in XMM-LSS
    '''

    counts_gals_rand = 0
    mass_sat_rand = []
    mass_sat_sf_rand = []
    mass_sat_q_rand = []
    n = 0
    num_p = 10.0  # number of blank pointings

    while n < int(num_p):  # n is number of blank pointings

        same_field = False
        while not same_field:
            id_rand = int(random()*len(cat_allobj_z_slice))
            ra_rand = cat_all_z_slice[id_rand]['RA']
            dec_rand = cat_all_z_slice[id_rand]['DEC']

            if ra_rand > 100 and ra_central > 100:
                same_field = True
            elif ra_rand < 100 and ra_central < 100:
                same_field = True
            else:
                same_field = False

        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(ra_rand, dec_rand, unit="deg"), coord_massive_gal, nthneighbor=1)

        if sep2d.degree > 1.0/dis/np.pi*180:  # make sure the blank pointing is away from any central galaxy
            coord_all_z_slice_rand = SkyCoord(cat_allobj_z_slice['RA']*u.deg, cat_allobj_z_slice['DEC']*u.deg)
            coord_rand = SkyCoord(ra_rand*u.deg, dec_rand*u.deg)
            cat_neighbors_rand = cat_all_z_slice[coord_all_z_slice_rand.separation(coord_rand).degree < 0.5/dis/np.pi*180]

            # cut the satellite mass catalog
            mass_neighbors_rand = cat_neighbors_rand['MASS_BEST']
            mass_neighbors_sf_rand = cat_neighbors_rand[cat_neighbors_rand['SSFR_BEST'] > -11]['MASS_BEST']
            mass_neighbors_q_rand = cat_neighbors_rand[cat_neighbors_rand['SSFR_BEST'] < -11]['MASS_BEST']

            #  total mass for satellites in blank pointings (all, sf & q)
            mass_sat_rand.append(np.sum(10 ** (mass_neighbors_rand[mass_neighbors_rand > 9] - 8)))  # unit 10**8 M_sun
            mass_sat_sf_rand.append(np.sum(10 ** (mass_neighbors_sf_rand[mass_neighbors_sf_rand > 9] - 8)))  # unit 10**8 M_sun
            mass_sat_q_rand.append(np.sum(10 ** (mass_neighbors_q_rand[mass_neighbors_q_rand > 9] - 8)))   # unit 10**8 M_sun

            # satellite counts in mass bins in blank pointings
            count_gal_rand, edges_rand = np.histogram(10 ** (mass_neighbors_rand - mass_cen), np.arange(0, 1.01, 0.1))
            counts_gals_rand += count_gal_rand

            n = n + 1

    total_mass_sat_rand = np.mean(mass_sat_rand)
    total_mass_sat_sf_rand = np.mean(mass_sat_sf_rand)
    total_mass_sat_q_rand = np.mean(mass_sat_q_rand)
    std_mass_sat_rand = np.std(mass_sat_rand)
    std_mass_sat_sf_rand = np.std(mass_sat_sf_rand)
    std_mass_sat_q_rand = np.std(mass_sat_q_rand)

    return counts_gals_rand/num_p, total_mass_sat_rand, total_mass_sat_sf_rand, total_mass_sat_q_rand,\
           std_mass_sat_rand, std_mass_sat_sf_rand,  std_mass_sat_q_rand

#################### MAIN FUNCTION ####################
total_mass_sat_log = open('total_mass_sat', 'w')  # record total mass of satellites for redshift evolution
for z in np.arange(0.3, 2.5, 0.1):
    print('============='+str(z)+'================')
    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal['ZPHOT'] - z) < 0.1]  # massive galaxies in this z slice
    cat_all_z_slice = cat_gal[abs(cat_gal['ZPHOT'] - z) < 0.1]  # all galaxies in this z slice

    dis = WMAP9.angular_diameter_distance(z).value  # angular diameter distance at redshift z
    search_r = 0.5  #Mpc

    # Fetch coordinated for massive gals
    cat_massive_z_slice['RA'].unit = u.deg
    cat_massive_z_slice['DEC'].unit = u.deg
    coord_massive_gal = SkyCoord.guess_from_table(cat_massive_z_slice)

    # search for central galaxies (exclude massive companions)
    sep_angle = -1.0  # init
    gal_mass = 99.0  #init
    cat_massive_z_slice_copy = np.copy(cat_massive_z_slice)
    for gal in cat_massive_z_slice_copy:
        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(gal['RA'], gal['DEC'], unit="deg"), coord_massive_gal, nthneighbor=2)
        if sep2d.degree[0] < 0.5/dis/np.pi*180:
            if round(sep2d.degree[0], 5) == round(sep_angle, 5):
                if gal['MASS_BEST'] > gal_mass:
                    cat_massive_z_slice = cat_massive_z_slice[cat_massive_z_slice['ID'] != id]
                else:
                    cat_massive_z_slice = cat_massive_z_slice[cat_massive_z_slice['ID'] != gal['ID']]

            sep_angle = sep2d.degree[0]
            gal_mass = gal['MASS_BEST']
            id = gal['ID']

    # counting neighbors for each massive central galaxy
    counts_gals = np.zeros(10)
    total_mass_sat = 0
    total_mass_sat_sf = 0
    total_mass_sat_q = 0
    massive_counts = len(cat_massive_z_slice)
    for gal in cat_massive_z_slice:

        # cut the z_slice catalog to get neighbors catalog
        cat_all_z_slice = cat_all_z_slice[cat_all_z_slice['ID'] != gal['ID']]
        coord_all_z_slice = SkyCoord(cat_all_z_slice['RA'] * u.deg, cat_all_z_slice['DEC'] * u.deg)
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)
        cat_neighbors = cat_all_z_slice[coord_all_z_slice.separation(coord_gal).degree < 0.5/dis/np.pi*180]

        if len(cat_neighbors) == 0:  # exlucde central gals which has no companion
            massive_counts -= 1
            continue

        mass_neighbors = cat_neighbors['MASS_BEST']
        if gal['MASS_BEST'] < max(mass_neighbors):  # exclude central gals which has larger mass companion
            massive_counts -= 1
            continue

        mass_neighbors_sf = cat_neighbors[cat_neighbors['SSFR_BEST'] > -11]['MASS_BEST']
        mass_neighbors_q = cat_neighbors[cat_neighbors['SSFR_BEST'] < -11]['MASS_BEST']
        mass_central = gal['MASS_BEST']

        # total mass of satellites (all, sf & q)
        total_mass_sat += np.sum(10**(mass_neighbors[mass_neighbors > 9] - 8))  # unit 10**8 M_sun
        total_mass_sat_sf += np.sum(10**(mass_neighbors_sf[mass_neighbors_sf > 9] - 8))  # unit 10**8 M_sun
        total_mass_sat_q += np.sum(10**(mass_neighbors_q[mass_neighbors_q > 9] - 8))  # unit 10**8 M_sun

        # satellite counts in mass bins
        count_gal, edges = np.histogram(10**(mass_neighbors - mass_central), np.arange(0, 1.01, 0.1))
        count_gal = np.array(count_gal, dtype='float64')

        # background/foreground correction
        # print(count_gal)
        count_gal_bkg, mass_sat_bkg, mass_sat_sf_bkg, mass_sat_q_bkg, std_mass_sat_bkg, std_mass_sat_sf_bkg, std_mass_sat_q_bkg = bkg(mass_central, gal['RA'], cat_all_z_slice)
        count_gal -= count_gal_bkg
        total_mass_sat -= mass_sat_bkg
        total_mass_sat_sf -= mass_sat_sf_bkg
        total_mass_sat_q -= mass_sat_q_bkg
        # print(count_gal_bkg)
        # print('...')
        counts_gals += count_gal

    ######### PLOT ###################
    # plot sat counts against mass fraction M/M_central

    fig = plt.figure(figsize=(7, 6))
    plt.rc('font', family='serif'), plt.rc('xtick', labelsize=15), plt.rc('ytick', labelsize=15)

    plt.errorbar(x=np.arange(0, 0.91, 0.1)+0.05, y=counts_gals/massive_counts, yerr=np.sqrt(counts_gals)/massive_counts, fmt='.k', markersize=16, capsize=3, elinewidth=1)
    plt.ylabel('neighbor counts', fontsize=16)
    plt.xlabel(r'$M_{sat}/M_{central}$', fontsize=16)
    plt.annotate('z='+str(z-0.1)+'~'+str(z+0.1)+', total:'+str(massive_counts), (1, 46), fontsize=14, xycoords='axes points')
    plt.annotate(r'total mass of satellites: '+str(round(total_mass_sat/massive_counts))+'$*10^8 M_{\odot}$', (1,31), fontsize=14, xycoords='axes points')
    plt.annotate(r'total mass of sf satellites: ' + str(round(total_mass_sat_sf / massive_counts)) + '$*10^8 M_{\odot}$', (1, 16), color='blue',fontsize=14, xycoords='axes points')
    plt.annotate(r'total mass of q satellites: ' + str(round(total_mass_sat_q / massive_counts)) + '$*10^8 M_{\odot}$', (1, 1), color='red', fontsize=14, xycoords='axes points')

    plt.yscale('log')
    plt.ylim([0.001, 5])
    plt.savefig('companion_counts_plots/'+str(z)+'.png')
    plt.close()

    # output total sat masses
    total_mass_sat_log.write(str(z) + ' ' + str(round(total_mass_sat / massive_counts)) +
                             ' '+ str(round(total_mass_sat_sf / massive_counts)) +
                             ' ' + str(round(total_mass_sat_q / massive_counts)) +
                             ' '+str(round(std_mass_sat_bkg) / massive_counts)+
                             ' '+str(round(std_mass_sat_sf_bkg) / massive_counts)+
                             ' '+str(round(std_mass_sat_q_bkg) / massive_counts)+
                             '\n')

total_mass_sat_log.close()