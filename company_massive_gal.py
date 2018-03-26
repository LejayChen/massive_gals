from astropy.table import Table
from astropy.cosmology import WMAP9
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.stats import bootstrap
from random import random
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

cat = Table.read('CUT2_CLAUDS_HSC_VISTA_Ks23.3_PHYSPARAM_TM.fits')
cat_gal = cat[cat['CLASS'] == 0]
cat_massive_gal = cat_gal[cat_gal['MASS_BEST'] > 11]


def bkg(mass_cen, ra_central, cat_all_z_slice_rand, coord_massive_gal_rand):
    '''
    return a background correction for each central galaxy by looking at black pointings without massive galaxies.
    mass_central: mass of the central galaxy,  ra_central: ra value for central galaxy. if ra_central>100, gal in COSMOS, if ra_central<100, gal in XMM-LSS
    '''
    # __init__
    counts_gals_rand = 0
    mass_sat_rand = []
    mass_sat_sf_rand = []
    n = 0
    num_p = 10  # number of blank pointings

    while n < num_p:  # get several blank pointings to estimate background
        same_field = False  # find a random pointing in the same field as the central galaxy
        while not same_field:
            id_rand = int(random()*len(cat_all_z_slice_rand))
            ra_rand = cat_all_z_slice_rand[id_rand]['RA'] + random()*1.2/dis/np.pi*180
            dec_rand = cat_all_z_slice_rand[id_rand]['DEC'] + random()*1.2/dis/np.pi*180
            if ra_rand > 100 and ra_central > 100:
                same_field = True
            elif ra_rand < 100 and ra_central < 100:
                same_field = True
            else:
                same_field = False

        idx, sep2d, dist3d = match_coordinates_sky(SkyCoord(ra_rand, dec_rand, unit="deg"), coord_massive_gal_rand, nthneighbor=1)
        if sep2d.degree > 2.5/dis/np.pi*180:  # make sure the random pointing is away from any central galaxy (blank)
            # cut all objects catalog to get neighbor catalog
            coord_all_z_slice_rand = SkyCoord(cat_all_z_slice_rand['RA']*u.deg, cat_all_z_slice_rand['DEC']*u.deg)
            coord_rand = SkyCoord(ra_rand*u.deg, dec_rand*u.deg)
            cat_neighbors_rand = cat_all_z_slice_rand[coord_all_z_slice_rand.separation(coord_rand).degree < 0.6/dis/np.pi*180]
            if len(cat_neighbors_rand) == 0: continue  # no gals (in masked region)

            # retrieve mass info in sat catalog (all, sf & q)
            mass_neighbors_rand = cat_neighbors_rand['MASS_BEST']
            mass_neighbors_sf_rand = cat_neighbors_rand[cat_neighbors_rand['SSFR_BEST'] > -11]['MASS_BEST']

            # calculate total mass for satellites in blank pointings (all, sf & q)
            mass_sat_rand.append(np.sum(10 ** (mass_neighbors_rand[mass_neighbors_rand > 10] - 8)))  # unit 10**8 M_sun
            mass_sat_sf_rand.append(np.sum(10 ** (mass_neighbors_sf_rand[mass_neighbors_sf_rand > 10] - 8)))  # unit 10**8 M_sun

            # calculate satellite counts in mass bins in blank pointings
            count_gal_rand, edges_rand = np.histogram(10 ** (mass_neighbors_rand - mass_cen), np.arange(0, 1.01, 0.1))
            counts_gals_rand += count_gal_rand

            n = n + 1

    # stats
    total_mass_sat_rand = np.mean(mass_sat_rand)  # mean value of total mass in an aperture
    total_mass_sat_sf_rand = np.mean(mass_sat_sf_rand)
    return mass_neighbors_rand, counts_gals_rand/float(num_p), total_mass_sat_rand, total_mass_sat_sf_rand

# ################### MAIN FUNCTION ################### #
total_mass_sat_log = open('total_mass_sat', 'w')  # record total mass of satellites for redshift evolution
for z in np.arange(0.9, 0.91, 0.1):
    print('============='+str(z)+'================')
    dis = WMAP9.angular_diameter_distance(z).value  # angular diameter distance at redshift z
    dis_l = WMAP9.comoving_distance(z - 0.1).value  # comoving distance at redshift z-0.1
    dis_h = WMAP9.comoving_distance(z + 0.1).value  # comoving distance at redshift z+0.1
    total_v = 4. / 3 * np.pi * (dis_h ** 3 - dis_l ** 3)  # Mpc^3
    survey_v = total_v * (4 / 41253.05)  # Mpc^3
    density = 0.00003  # desired constant (cumulative) volume number density (Mpc^-3)
    num = int(density * survey_v)

    cat_massive_z_slice = cat_massive_gal[abs(cat_massive_gal['ZPHOT'] - z) < 0.1]  # massive galaxies in this z slice
    cat_massive_z_slice.sort('MASS_BEST')
    cat_massive_z_slice.reverse()
    cat_massive_z_slice = cat_massive_z_slice[:num]  # select most massive ones (keep surface density constant in different redshift bins)

    cat_all_z_slice = cat_gal[abs(cat_gal['ZPHOT'] - z) < 0.2]  # all galaxies in this z slice

    # Fetch coordinates for massive gals
    cat_massive_z_slice['RA'].unit = u.deg
    cat_massive_z_slice['DEC'].unit = u.deg
    coord_massive_gal = SkyCoord.guess_from_table(cat_massive_z_slice)

    # counting neighbors and calculating total mass of neighbors for each massive central galaxy
    # __init__
    counts_gals = np.zeros(10)
    massive_counts = len(cat_massive_z_slice)
    SMF_z = np.zeros(50)
    SMF_z_bkg = np.zeros(50)
    mass_sat = []  # contains mass of sats
    mass_sat_sf = []
    cylinder_length = 1.5 * 0.03 * (1 + z)  # half cylinder height centered at one central galaxy

    for gal in cat_massive_z_slice:
        # cut the z_slice catalog to get neighbors catalog
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)

        cat_neighbors_z_slice = cat_all_z_slice[abs(cat_all_z_slice['ZPHOT'] - gal['ZPHOT']) < cylinder_length]
        coord_neighbors_z_slice = SkyCoord(cat_neighbors_z_slice['RA'] * u.deg, cat_neighbors_z_slice['DEC'] * u.deg)
        cat_neighbors = cat_neighbors_z_slice[coord_neighbors_z_slice.separation(coord_gal).degree < 0.6/dis/np.pi*180]
        cat_neighbors = cat_neighbors[cat_neighbors['ID'] != gal['ID']]  # exclude central gal itself

        if len(cat_neighbors) == 0:  # exclude central gals which has no companion
            massive_counts -= 1
            continue

        # retrieve mass info in sat catalog (all, sf & q)
        mass_neighbors = cat_neighbors['MASS_BEST']
        if gal['MASS_BEST'] < max(cat_neighbors['MASS_BEST']):  # exclude central gals which has larger mass companion
            massive_counts -= 1
            continue
        mass_neighbors_sf = cat_neighbors[cat_neighbors['SSFR_BEST'] > -11]['MASS_BEST']
        mass_central = gal['MASS_BEST']

        # ### satellite counts in mass bins
        # background subtracted

        count_gal, edges = np.histogram(10**(mass_neighbors - mass_central), np.arange(0, 1.01, 0.1))
        mass_neighbors_bkg, count_gal_bkg, mass_sat_bkg, mass_sat_sf_bkg = bkg(mass_central, gal['RA'], np.copy(cat_neighbors_z_slice), coord_massive_gal)
        counts_gals = counts_gals + (np.array(count_gal, dtype='float64') - count_gal_bkg)  # total sat_gal counts (binned with mass fraction)
        # SMF_z += np.histogram(mass_neighbors, bins=50, range=(10., 12.))[0]
        # SMF_z_bkg += np.histogram(mass_neighbors_bkg, bins=50, range=(10., 12.))[0]

        # ### total mass of satellites (all, sf & q; all=sf+q)
        # background subtracted
        mass_sat.append(np.sum(10 ** (mass_neighbors[mass_neighbors > 10] - 8)) - mass_sat_bkg)  # unit 10**8 M_sun
        mass_sat_sf.append(np.sum(10 ** (mass_neighbors_sf[mass_neighbors_sf > 10] - 8)) - mass_sat_sf_bkg)  # unit 10**8 M_sun

    # plt.plot(np.arange(10, 12, 0.04), SMF_z)
    # plt.plot(np.arange(10, 12, 0.04), SMF_z_bkg)
    # plt.show()

    # averaged total mass of satellites
    # bootstrap sample to get variation on total mass of satellites
    bootsresult_sat = bootstrap(np.array(mass_sat), massive_counts, bootfunc=np.mean)
    bootsresult_sat_sf = bootstrap(np.array(mass_sat_sf), massive_counts, bootfunc=np.mean)

    total_mass_sat = np.mean(bootsresult_sat)
    total_mass_sat_sf = np.mean(bootsresult_sat_sf)
    total_mass_sat_q = total_mass_sat - total_mass_sat_sf

    std_mass_sat_total = np.std(bootsresult_sat)
    std_mass_sat_sf_total = np.std(bootsresult_sat_sf)
    std_mass_sat_q_total = np.sqrt(std_mass_sat_total ** 2 + std_mass_sat_sf_total ** 2)

    # ######## PLOT #########
    # plot sat counts against mass fraction M/M_central
    np.set_printoptions(precision=4)
    fig = plt.figure(figsize=(7, 6))
    plt.rc('font', family='serif'), plt.rc('xtick', labelsize=15), plt.rc('ytick', labelsize=15)
    plt.errorbar(x=np.arange(0, 0.91, 0.1)+0.05, y=counts_gals/massive_counts, yerr=np.sqrt(counts_gals)/massive_counts, fmt='.k', markersize=16, capsize=3, elinewidth=1)
    plt.ylabel('neighbor counts', fontsize=16)
    plt.xlabel(r'$M_{sat}/M_{central}$', fontsize=16)
    plt.annotate('z='+str(round(z-0.1,2))+'~'+str(round(z+0.1,2))+', total:'+str(massive_counts), (1, 46), fontsize=14, xycoords='axes points')
    plt.annotate(r'total mass of satellites: '+str(round(total_mass_sat))+'$*10^8 M_{\odot}$', (1,31), fontsize=14, xycoords='axes points')
    plt.annotate(r'total mass of sf satellites: ' + str(round(total_mass_sat_sf)) + '$*10^8 M_{\odot}$', (1, 16), color='blue',fontsize=14, xycoords='axes points')
    plt.annotate(r'total mass of q satellites: ' + str(round(total_mass_sat_q)) + '$*10^8 M_{\odot}$', (1, 1), color='red', fontsize=14, xycoords='axes points')
    plt.title(str(counts_gals/massive_counts), fontsize=14)
    plt.yscale('log')
    plt.ylim([0.0005, 10])
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