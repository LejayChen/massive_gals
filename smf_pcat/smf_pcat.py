import sys, os
from random import random
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.cosmology import *
from astropy.stats import bootstrap
from astropy.table import Table, Column
from mpi4py import MPI

def check_edge(ra_rand, dec_rand):
    cat_random_cut = cat_random[abs(cat_random['ra'] - ra_rand) < r_iso / dis / np.pi * 180]
    cat_random_cut = cat_random_cut[abs(cat_random_cut['dec'] - dec_rand) < r_iso / dis / np.pi * 180]
    try:
        ra_ratio = len(cat_random_cut[cat_random_cut['ra']<ra_rand])/len(cat_random_cut[cat_random_cut['ra']>ra_rand])
        dec_ratio = len(cat_random_cut[cat_random_cut['dec']<dec_rand])/len(cat_random_cut[cat_random_cut['dec']>dec_rand])
    except (ValueError, ZeroDivisionError):
        return True

    if ra_ratio > 1/0.75 or ra_ratio < 0.75:
        return True
    elif dec_ratio > 1/0.75 or dec_ratio < 0.75:
        return True
    else:
        return False

def bkg_all_area(cat_gal_non_proximity_zslice, area_total):
    counts_gals_rand = np.histogram(cat_gal_non_proximity_zslice[mass_keyname], weights=weights, bins=bin_edges)[0]
    area_total_non_proximity = cat_random_bkg_pos/len(cat_random_nomask)*area_total
    area_cen_aperture = np.pi*(r_high**2-r_low**2)/ (dis * np.pi / 180)**2

    return counts_gals_rand * area_cen_aperture / area_total_non_proximity

def bkg(cat_random_bkg_pos, cat_neighbors_z_slice_rand, cat_isolated_zgal_cut, mass_cen, count_bkg, n_per_central=1, bkg_size_factor = 1.5):
    global z, overlapping_factor

    cat_neighbors_z_slice_rand = cat_neighbors_z_slice_rand[np.logical_and(cat_neighbors_z_slice_rand[mass_keyname] > masscut_low,cat_neighbors_z_slice_rand[mass_keyname] < masscut_high)]
    
    if count_bkg <= 1:
        max_iter = int(len(cat_random_bkg_pos)/3)
    elif count_bkg <=3:
        max_iter = int(len(cat_random_bkg_pos)/6)
    elif count_bkg <=10:
        max_iter = int(len(cat_random_bkg_pos)/15)
    else:
        max_iter = 5

    # randomize bkg positions
    shuffle_rand_pos_ids = np.arange(len(cat_random_bkg_pos))
    np.random.shuffle(shuffle_rand_pos_ids)

    n = 0
    flag_bkg = 0
    num_before_success = 0
    counts_gals_rand = np.zeros(bin_number)
    while n < n_per_central:  # get several blank pointing's to estimate background
        id_rand = shuffle_rand_pos_ids[num_before_success]
        ra_rand = cat_random_bkg_pos[id_rand]['ra']
        dec_rand = cat_random_bkg_pos[id_rand]['dec']
        coord_rand = SkyCoord(ra_rand, dec_rand, unit="deg")

        num_before_success += 1
        if num_before_success > max_iter:
            flag_bkg = 1
            break

        if check_edge(ra_rand, dec_rand):
            continue

        # test if bkg position is celan of isolated massive galaxies
        if len(cat_isolated_zgal_cut)>0:
            ra_min_sep = min(abs(ra_rand - cat_isolated_zgal_cut['RA']))
            dec_min_sep = min(abs(ra_rand - cat_isolated_zgal_cut['DEC']))
        else:
            ra_min_sep = 10
            dec_min_sep = 10
            
        if cat_name == 'ELAIS_deep':
            sph_coord_factor = 2
        else:
            sph_coord_factor = 1.1

        if ra_min_sep>overlapping_factor*(1+1/bkg_size_factor)*sph_coord_factor*r_iso*convert_factor and dec_min_sep>overlapping_factor*(1+1/bkg_size_factor)*r_iso*convert_factor:
            bkg_clean = True
        else:
            coord_isolated_gal_rand = SkyCoord(cat_isolated_zgal_cut['RA'] * u.deg, cat_isolated_zgal_cut['DEC'] * u.deg)
            idx, sep2d, dist3d = match_coordinates_sky(coord_rand, coord_isolated_gal_rand, nthneighbor=1)
            if sep2d.degree > overlapping_factor*(1+1/bkg_size_factor)*r_iso*convert_factor:
                bkg_clean = True
            else:
                bkg_clean = False
            
        if bkg_clean:
            cat_neighbors_rand = cat_neighbors_z_slice_rand[np.logical_and(abs(cat_neighbors_z_slice_rand['RA'] - ra_rand) < r_iso/bkg_size_factor*sph_coord_factor*convert_factor, abs(cat_neighbors_z_slice_rand['DEC'] - dec_rand) < r_iso/bkg_size_factor*convert_factor )]

            coord_neighbors_rand = SkyCoord(cat_neighbors_rand['RA'] * u.deg, cat_neighbors_rand['DEC'] * u.deg)
            if len(cat_neighbors_rand) == 0:
                continue

            # choose radial range
            if r_low>0.0:
                cat_neighbors_rand = cat_neighbors_rand[np.logical_and(coord_neighbors_rand.separation(coord_rand).degree < r_high/bkg_size_factor*convert_factor, coord_neighbors_rand.separation(coord_rand).degree > r_low/bkg_size_factor*convert_factor)]
            else:
                cat_neighbors_rand = cat_neighbors_rand[coord_neighbors_rand.separation(coord_rand).degree < r_high/bkg_size_factor*convert_factor]

            if len(cat_neighbors_rand) == 0:
                continue
            elif max(cat_neighbors_rand[mass_keyname])>masscut_host_high:
                continue
            else:
                mass_neighbors_rand = cat_neighbors_rand[mass_keyname]

                if ssfq == 'all':
                    sfq_weights_rand = np.ones(len(mass_neighbors_rand))
                elif ssfq == 'ssf':
                    sfq_weights_rand = cat_neighbors_rand[sfq_keyname]
                else:
                    sfq_weights_rand = 1 - cat_neighbors_rand[sfq_keyname]

                # weights = np.array(sfq_weights_rand / completeness_est(mass_neighbors_rand, cat_neighbors_rand[sfq_keyname], z))
                weights = np.array(sfq_weights_rand)
                if not rel_mscale:
                    counts_gals_rand += np.histogram(mass_neighbors_rand, weights=weights, bins=bin_edges)[0]
                    # print('bkg gal mass max',max(mass_neighbors_rand))
                else:
                    rel_mass_neighbors_rand = mass_neighbors_rand - mass_cen
                    counts_gals_rand += np.histogram(rel_mass_neighbors_rand, weights=weights, bins=rel_bin_edges)[0]
                
            n = n + 1

    # print(num_before_success)
    if n>0:
        return num_before_success, counts_gals_rand*bkg_size_factor**2/n, flag_bkg
    else:
        return num_before_success, counts_gals_rand, flag_bkg


def completeness_est(mass_list, sfProb_list, z):
    try:
        completeness_sf = np.genfromtxt('../mass_completeness_data/allFields_' + str(round(z - 0.1, 1)) + '_z_' + str(round(z + 0.1,1)) + '_sf_nopert_nan.txt')
        completeness_q = np.genfromtxt('../mass_completeness_data/allFields_' + str(round(z - 0.1, 1)) + '_z_' + str(round(z + 0.1,1)) + '_q_nopert_nan.txt')
        completeness = np.array([])
        for idx in range(len(mass_list)):
            if sfProb_list[idx] > 0.5:
                completeness = np.append(completeness, np.interp(mass_list[idx], completeness_sf[0], completeness_sf[3]))
            else:
                completeness = np.append(completeness, np.interp(mass_list[idx], completeness_q[0], completeness_q[3]))

        completeness[np.isnan(completeness)] = 1.
        return completeness
    except:
        return np.ones(len(mass_list))

def Ms_to_r200(log_Ms,z): 
    rho_bar = 9.9e-30  # g/cm^3
    Ms = 10**log_Ms # solar masses
    if z<0.48:
        M1 = 10 ** 12.52 # solar masses
        Ms0 = 10 ** 10.916 # solar masses
        beta = 0.457
        delta = 0.566
        gamma = 1.53
    elif z>=0.48 and z<0.74:
        M1 = 10 ** 12.725 # solar masses
        Ms0 = 10 ** 11.038 # solar masses
        beta = 0.466
        delta = 0.61
        gamma = 1.95
    elif z>=0.74:
        M1 = 10 ** 12.722 # solar masses
        Ms0 = 10 ** 11.1 # solar masses
        beta = 0.47
        delta = 0.393
        gamma = 2.51
    log_mh = np.log10(M1) + beta * np.log10(Ms / Ms0) + (Ms / Ms0) ** delta / (1 + (Ms / Ms0) ** (-1 * gamma)) - 0.5 # Leauthaud+2012
    r200 = ((3*10**log_mh*1.989e30*1e3)/(800*np.pi*rho_bar))**(1/3)/3.086e21 # in kpc
    return r200/1000  # in Mpc

def Ms_to_r200_no2(log_Ms,z_min,z_max):
    rho_bar = 9.9e-30  # g/cm^3
    data = np.genfromtxt('SHMR_total_z-'+str(round((z_min+z_max)/2,2))+'.dat') # from Shuntov et al. 2022 A&A, 664 (2022) A61 
    data = data[1:,]
    log_Mh_list = np.log10(data[:,0])
    log_Ms_list = np.log10(data[:,1])
    log_Mh = np.interp(log_Ms,log_Ms_list,log_Mh_list)
    
    r200 = ((3*10**log_Mh*1.989e30*1e3)/(4*np.pi*200*rho_bar))**(1/3)/3.086e21 # in kpc
    return r200/1000  # in Mpc

# multi-threading settings
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nProcs = comm.Get_size()

# cosmology ##############
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# ################# START #####################
correct_masked = True
save_results = True
save_catalogs = False
small_size_test = False
alt_r200 = False

ssfq = sys.argv[1]
z_min = eval(sys.argv[2])
z_max= eval(sys.argv[3])
path = sys.argv[4]
scratch_path = '/scratch/lejay/'+path 
bin_number = eval(sys.argv[5])
boot_num = int(eval(sys.argv[6]))

sigma_z = 0.06
masscut_low = eval(sys.argv[7])
masscut_high = 13.0
masscut_host = eval(sys.argv[8])
masscut_host_high = eval(sys.argv[9])
bin_edges = np.linspace(masscut_low, masscut_high, num=bin_number+1)
rel_bin_edges = np.linspace(-4, 0, num=bin_number+1)
bkg_size_factor = 1.5 # radius X times smaller for bkg selection 


overlapping_factor = eval(sys.argv[10]) # closest distance factor a random position is to a massive galaxy (1=r_iso)
csfq = sys.argv[11]  # csf, cq, all

# radius of counting satellites
r_low = eval(sys.argv[12])  # Mpc
r_high = eval(sys.argv[13])  # Mpc
if r_high>= 0.7 or masscut_host<11.0:
    r_iso = r_high  # Mpc (isolation criteria radius)
else:
    r_iso = 0.7

overlap_exclude = bool(eval(sys.argv[14]))
rel_mscale = bool(eval(sys.argv[15])) # relative mass scale
rel_riso = bool(eval(sys.argv[16])) # relative aperture size
r_factor = eval(sys.argv[17]) # only use satellites within  r_factor*r200
sat_z_cut = eval(sys.argv[18])
evo_mcen =  bool(eval(sys.argv[19]))
if rel_riso and masscut_host_high>12.0: # R200 maybe unreasonably to large for M>12.0
    masscut_host_high = 12.0

print('relative scales status', rel_mscale, rel_riso)

if not overlap_exclude:
    overlapping_factor = 0.0

if path[-1] != '/' and save_results:
    raise NameError('path is not a directory!')
elif save_results:
    print('will save results to ' + scratch_path)
else:
    print('will NOT save results!')


cat_names = ['COSMOS_deep','DEEP_deep','ELAIS_deep','XMM-LSS_deep']
cat_name = cat_names[rank % len(cat_names)]
cores_per_cat = nProcs // len(cat_names)
rank_in_cat = rank // len(cat_names)

catalog_path = '/home/lejay/projects/def-sawicki/lejay/phys_added_pcats/'
cat = Table.read(catalog_path + cat_name.replace('_deep','') + '_galaxies_241214_trimmed.fits')  # read-in
zkeyname = 'Z_COMBINE'
mass_keyname = 'MASS_MED'
sfq_keyname = 'sfq_nuvrk_onebin_fc'
id_keyname = 'ID'
ra_key = 'RA'
dec_key = 'DEC'

if 'XMM' not in cat_name:
    cat = cat[cat['inside_u'] == True]
else:
    cat = cat[cat['inside_uS_deep'] == True]
    cat = cat[cat['inside_j'] == True]
cat = cat[cat['inside_hsc'] == True]
cat = cat[cat['isStar']==False]
cat = cat[cat['inside_hsc'] == True]
cat = cat[cat['isCompact']==False]
cat = cat[cat['i_compact_flag']==False]
cat = cat[cat['snr_i']>3]
cat = cat[cat['snr_r']>1.5]
cat = cat[cat['snr_z']>1.5]
cat_gal = cat[cat['i_cmodel']>0]
print(cat_name,len(cat_gal))

cat_gal = cat_gal[cat_gal[zkeyname] > 0]
cat_gal = cat_gal[cat_gal[zkeyname] > z_min - 2*0.06*sat_z_cut*(1+z_min)]
cat_gal = cat_gal[cat_gal[zkeyname] < z_max + 2*0.06*sat_z_cut*(1+z_max)]
cat_gal = cat_gal[cat_gal[sfq_keyname] >= 0.0]
cat_gal = cat_gal[cat_gal[sfq_keyname] <= 1.0]
cat_gal = cat_gal[cat_gal[mass_keyname] >= masscut_low]
cat_gal = cat_gal[cat_gal[mass_keyname] < masscut_high]
del cat

# read in random point catalog
cat_random = Table.read('/home/lejay/random_point_cat/'+cat_name + '_random_point.fits')
cat_random = cat_random[cat_random['inside_hsc'] == 0]
if cat_name == 'XMM-LSS_deep':
    cat_random = cat_random[cat_random['inside_uS_deep'] == 0]
    cat_random = cat_random[cat_random['inside_j'] == 0]
else:
    cat_random = cat_random[cat_random['inside_u'] == 0]
cat_random_nomask = np.copy(cat_random)
cat_random = cat_random[cat_random['MASK'] != 0]
area_total_list = [5.19, 4.27, 4.51, 2.63] # 'COSMOS_deep','DEEP_deep','ELAIS_deep','XMM-LSS_deep'
area_total = area_total_list[rank % len(cat_names)]

# trim cat_gal
cat_gal.remove_columns(['snr_i','snr_r','snr_z','i_cmodel','isStar','inside_hsc','isCompact','i_compact_flag'])
print(z_min,z_max,cat_name,rank_in_cat,len(cat_gal))

#mass correction with COSMOS2020
if cat_name != 'XMM-LSS_deep':
    mass_offset = np.load('mass_err/mass_err_median_all_nonir_'+str(z_min)+'_'+str(z_max)+'.npy')
else:
    mass_offset = np.load('mass_err/mass_err_median_all_noirac_'+str(z_min)+'_'+str(z_max)+'.npy')

mass_offset_x = mass_offset[0]
mass_offset_y = mass_offset[1] 
p = np.polyfit(mass_offset_x,mass_offset_y,deg=3) # linear fit
for gal in cat_gal:
    # correction by m_myrun - m_c20 mass offset
    if gal[mass_keyname] > 8 and gal[mass_keyname] <= 12.2:
        m_corr = p[0]*gal[mass_keyname]**3+p[1]*gal[mass_keyname]**2+p[2]*gal[mass_keyname]+p[3]
        if gal[mass_keyname]>=mass_offset_x[0]:
            gal[mass_keyname] = gal[mass_keyname] - m_corr
        else:
            gal[mass_keyname] = gal[mass_keyname] - (p[0]*mass_offset_x[0]**3+p[1]*mass_offset_x[0]**2+p[2]*mass_offset_x[0]+p[3])  

#bootstrap resampling
smf_dist_arr = np.zeros(bin_number)
smf_dist_bkg_arr = np.zeros(bin_number)
mass_key_ori = cat_gal[mass_keyname].copy()  # original
z_key_ori = cat_gal[zkeyname].copy()  # original
mass_centrals_ori = []
isolated_counts_ori = 0
count_bkg_ori = 0
for boot_iter in range(boot_num):
    print('')
    print(cat_name, 'boot_iter '+str(boot_iter)+' started.')
    
    if boot_iter > 0:  # number of iteration of bootstrapping
        cat_gal[mass_keyname] = mass_key_ori
        cat_gal[zkeyname] = z_key_ori
        # z_scatter = np.random.normal(cat_gal[zkeyname], sigma_z * (cat_gal[zkeyname] + 1))  # photoz scatter
        # mass_scatter = np.log10(abs(np.random.normal(10 ** (cat_gal[mass_keyname] - 10),
        #                   (cat_gal['MASS_SUP'] - cat_gal['MASS_INF'])/2 * 10 ** (cat_gal['MASS_MED'] - 10)))) + 10  # mass scatter

        # cat_gal[mass_keyname] = mass_scatter
        # cat_gal[zkeyname] = z_scatter
        boot_idx = bootstrap(np.arange(len(cat_gal)), bootnum=1)
        cat_gal_copy = cat_gal[boot_idx[0].astype(int)]
    else:
        cat_gal_copy = cat_gal

    # select massive galaxies
    if not evo_mcen:
        cat_massive_gal = cat_gal_copy[cat_gal_copy[mass_keyname] > masscut_host] # lower mass cut
    else:
        cat_massive_gal = cat_gal_copy[cat_gal_copy[mass_keyname] > masscut_host - 0.16*z_max] # lower mass cut

    cat_massive_z_slice = cat_massive_gal[np.logical_and(cat_massive_gal[zkeyname]>z_min-2.0 * sat_z_cut * sigma_z * (1 + z_min), cat_massive_gal[zkeyname]<z_max+2.0 * sat_z_cut * sigma_z * (1 + z_max))]
    print('massive gals rank_in_cat:', rank_in_cat, len(cat_massive_z_slice), max(cat_massive_z_slice[mass_keyname]))

    #select isolated massive galaxies
    # cat_gal_non_proximity = cat_gal['ID',zkeyname, mass_keyname,'RA','DEC']
    isolated_ids = []
    for gal_idx, gal in enumerate(cat_massive_z_slice):  

        if rel_riso:
            if not alt_r200:
                r_iso =  Ms_to_r200(gal[mass_keyname],gal[zkeyname])
            else:
                r_iso =  Ms_to_r200_no2(gal[mass_keyname],z_min,z_max)

        dis = cosmo.angular_diameter_distance(gal[zkeyname]).value
        convert_factor = 1 / dis / np.pi * 180 # convert Mpc to degree (at the redshift of central galaxy)
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)
        cat_neighbors_massive_z_slice = cat_massive_gal[abs(cat_massive_gal[zkeyname] - gal[zkeyname]) < sat_z_cut * sigma_z * (1 + gal[zkeyname])]
        
        if cat_name == 'ELAIS_deep':
            sph_coord_factor = 2
        else:
            sph_coord_factor = 1.1
        cat_neighbors_massive = cat_neighbors_massive_z_slice[np.logical_and(abs(cat_neighbors_massive_z_slice['RA'] - gal['RA']) < sph_coord_factor * r_iso * convert_factor, abs(cat_neighbors_massive_z_slice['DEC'] - gal['DEC']) < r_iso * convert_factor)]

        # #### spatial selection
        coord_neighbors_massive = SkyCoord(cat_neighbors_massive['RA'] * u.deg, cat_neighbors_massive['DEC'] * u.deg)
        cat_neighbors_massive = cat_neighbors_massive[coord_neighbors_massive.separation(coord_gal).degree < r_iso * convert_factor]

        # evolving mass cut on central
        if evo_mcen:
            if gal[mass_keyname] < masscut_host - 0.16*gal[zkeyname]:
                continue
            elif gal[mass_keyname] > masscut_host_high - 0.16*gal[zkeyname]:
                continue

        # isolation cut on central
        if len(cat_neighbors_massive) == 0:  # central gals which has no companion
            continue
        
        elif gal[mass_keyname] < max(cat_neighbors_massive[mass_keyname]):  # no more-massive companions
            # print('more massive nearby',gal[mass_keyname], max(cat_neighbors_massive[mass_keyname]))
            continue
        else:
            isolated_ids.append(gal_idx)

    # isolated massive galaxy sample
    cat_isolated_z_slice_allmass = cat_massive_z_slice[isolated_ids]
    cat_isolated_z_slice = cat_isolated_z_slice_allmass[np.logical_and(cat_isolated_z_slice_allmass[zkeyname]>z_min, cat_isolated_z_slice_allmass[zkeyname]<z_max)]
    cat_isolated_z_slice = cat_isolated_z_slice[cat_isolated_z_slice[mass_keyname]<masscut_host_high]

    cat_massive_z_slice = cat_massive_z_slice[cat_massive_z_slice[mass_keyname]<masscut_host_high]
    cat_massive_z_slice = cat_massive_gal[np.logical_and(cat_massive_gal[zkeyname]>z_min, cat_massive_gal[zkeyname]<z_max)]
    print('isolated gals:', len(cat_isolated_z_slice), 'massive gals',len(cat_massive_z_slice), min(cat_isolated_z_slice[mass_keyname]), max(cat_isolated_z_slice[mass_keyname]))

    # split massive gal sample (for each node)
    if rank_in_cat != cores_per_cat-1:
        cat_isolated_z_slice = cat_isolated_z_slice[rank_in_cat*int(len(cat_isolated_z_slice)/(nProcs/len(cat_names))) : (rank_in_cat+1)*int(len(cat_isolated_z_slice)/(nProcs/len(cat_names)))]
    else:
        cat_isolated_z_slice = cat_isolated_z_slice[rank_in_cat*int(len(cat_isolated_z_slice)/(nProcs/len(cat_names))) : (rank_in_cat+1)*int(len(cat_isolated_z_slice)/(nProcs/len(cat_names)))]

    # set number of bkg per central
    if len(cat_isolated_z_slice)<40:
        n_per_central = 2
    else:
        n_per_central = 1

    # small scale test check
    if small_size_test:
        cat_isolated_z_slice_torun = cat_isolated_z_slice[np.random.choice(np.arange(len(cat_isolated_z_slice)), size=min(len(cat_isolated_z_slice),50), replace=False)]
    else:
        cat_isolated_z_slice_torun = cat_isolated_z_slice
    
    print('===rank:'+str(rank)+'===rank_in_cat=' + str(rank_in_cat) + '===z_min=' + str(round(z_min, 1)) + '===z_max=' + str(round(z_max, 1))+ '====='+cat_name+'==='+str(len(cat_isolated_z_slice))+'===')
    print('isolated gals, rank_in_cat:', rank_in_cat, len(cat_isolated_z_slice_torun), min(cat_isolated_z_slice_torun[mass_keyname]),max(cat_isolated_z_slice_torun[mass_keyname]))

    # CORE CALCULATION
    isolated_counts = 0
    smf_dist = np.zeros(bin_number)
    smf_dist_bkg = np.zeros(bin_number)
    mass_centrals = []
    count_bkg = 0
    dM_sat = [] # difference of mass between central and most massive satellite (no bkg subtraction here)
    n_sat = []
    for gal in cat_isolated_z_slice_torun:  # [np.random.randint(len(cat_isolated_z_slice), size=300)]:

        if rel_riso:
            if not alt_r200:
                r_iso =  Ms_to_r200(gal[mass_keyname],gal[zkeyname])
            else:
                r_iso =  Ms_to_r200_no2(gal[mass_keyname],z_min,z_max)
            r_high = r_iso*r_factor

        # cut on central SF/Q
        if csfq != 'all':
            if csfq == 'csf' and gal[sfq_keyname] < 0.5:
                continue
            elif csfq == 'cq' and gal[sfq_keyname] >= 0.5:
                continue

        dis = cosmo.angular_diameter_distance(gal[zkeyname]).value
        convert_factor = 1 / dis / np.pi * 180 # convert Mpc to degree (at the redshift of central galaxy)
        coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)

        # prepare neighbors catalog
        cat_neighbors_z_slice = cat_gal_copy[abs(cat_gal_copy[zkeyname] - gal[zkeyname]) < sat_z_cut * sigma_z * (1 + gal[zkeyname])]

        if cat_name == 'ELAIS_deep':
            sph_coord_factor = 2
        else:
            sph_coord_factor = 1.1

        cat_neighbors = cat_neighbors_z_slice[np.logical_and(abs(cat_neighbors_z_slice['RA'] - gal['RA']) < sph_coord_factor * r_iso * convert_factor, abs(cat_neighbors_z_slice['DEC'] - gal['DEC']) < r_iso * convert_factor )]

        # choose sats within r_high
        coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
        cat_neighbors = cat_neighbors[coord_neighbors.separation(coord_gal).degree < r_high * convert_factor]
        cat_neighbors = cat_neighbors[cat_neighbors['ID'] != gal['ID']]
        if len(cat_neighbors) < 1:
            print('excluded: not enough satellites', len(cat_neighbors), gal[mass_keyname])
            dM_sat.append(0)
            n_sat.append(0)
            continue
        elif gal[mass_keyname] < max(cat_neighbors[mass_keyname]):  # central gals which has no companion
            print('excluded: higher mass satellite', len(cat_neighbors), gal[mass_keyname], max(cat_neighbors[mass_keyname]))
            dM_sat.append(0)
            n_sat.append(0)
            continue

        # exclude sats within r_low
        if r_low > 0.0:
            coord_neighbors = SkyCoord(cat_neighbors['RA'] * u.deg, cat_neighbors['DEC'] * u.deg)
            cat_neighbors = cat_neighbors[coord_neighbors.separation(coord_gal).degree >= r_low * convert_factor]
            if len(cat_neighbors) == 0:  # central gals which has no companion
                dM_sat.append(0)
                n_sat.append(0)
                continue

        # statistics #
        isolated_counts += 1
        mass_centrals.append(gal[mass_keyname])
        dM_sat.append(gal[mass_keyname] - max(cat_neighbors[mass_keyname]))
        n_sat.append(len(cat_neighbors))

        if ssfq == 'all':
            sfq_weights = np.ones(len(cat_neighbors))
        elif ssfq == 'ssf':
            sfq_weights = np.array(cat_neighbors[sfq_keyname])
        else:
            sfq_weights = np.array(1 - cat_neighbors[sfq_keyname]) 

        # sat_weights = np.array(sfq_weights/completeness_est(mass_neighbors, cat_neighbors[sfq_keyname], z))
        sat_weights = np.array(sfq_weights)

        #absolute / relative mass scale
        mass_neighbors = cat_neighbors[mass_keyname]
        if not rel_mscale:
            count_binned = np.histogram(mass_neighbors, weights=sat_weights, bins=bin_edges)[0]
        else:
            rel_mass_neighbors = mass_neighbors - gal[mass_keyname]
            count_binned = np.histogram(rel_mass_neighbors, weights=sat_weights, bins=rel_bin_edges)[0]

        sat_counts = np.array(count_binned, dtype='f8')
        smf_dist += sat_counts 

        # count galaxies with M>9.5
        mass_neighbors_select = mass_neighbors[mass_neighbors>9.5]
        print(gal['ID'],'n_sat_tot',len(mass_neighbors),'n_sat_9.5',len(mass_neighbors_select))
        # print(gal['ID'],sum(sat_counts), gal[mass_keyname], max(cat_neighbors[mass_keyname]))

        # positions of central gals to avoid in bkg
        cat_isolated_zgal_cut = cat_isolated_z_slice_allmass[cat_isolated_z_slice_allmass[mass_keyname]>gal[mass_keyname]]
        cat_isolated_zgal_cut = cat_isolated_zgal_cut[abs(cat_isolated_zgal_cut[zkeyname] - gal[zkeyname]) < 2.0 * sat_z_cut * sigma_z * (1 + gal[zkeyname])]
        cat_isolated_zgal_cut =  cat_isolated_zgal_cut['ID', mass_keyname, zkeyname, 'RA', 'DEC']
        num_before_success, sat_bkg, flag_bkg = bkg(cat_random_nomask, cat_neighbors_z_slice, cat_isolated_zgal_cut, gal[mass_keyname], count_bkg, n_per_central=n_per_central)
        if flag_bkg == 0:  # success
            count_bkg += 1
            smf_dist_bkg += sat_bkg
            # print(gal['ID'],sum(sat_counts), gal[mass_keyname],max(cat_neighbors[mass_keyname]), sum(sat_bkg), num_before_success,len(cat_isolated_zgal_cut))

        # secound bkg method
        # cat_gal_non_proximity_zslice = cat_gal_non_proximity[(cat_gal_non_proximity[zkeyname] - gal[zkeyname]) < sat_z_cut * sigma_z * (1 + gal[zkeyname])]
        # sat_bkg = bkg_all_area(cat_gal_non_proximity_zslice, area_total)

    # add results from this bootstrap iteration
    if count_bkg==0:
        print(rank, rank_in_cat, 'count bkg == 0')

    print('bkg/central counts', count_bkg, isolated_counts, len(cat_isolated_z_slice_torun))
    smf_dist_bkg = smf_dist_bkg / float(count_bkg) * isolated_counts

    # stack results from different iterations
    if isolated_counts > 0:
        smf_dist_arr = np.vstack((smf_dist_arr, smf_dist))
        smf_dist_bkg_arr = np.vstack((smf_dist_bkg_arr, smf_dist_bkg))
        if boot_iter == 0:
            mass_centrals_ori = mass_centrals
            isolated_counts_ori = isolated_counts
            count_bkg_ori = count_bkg

        # write isolated galaxy catalog to file
        if boot_iter == 0 and (csfq == 'all' and ssfq == 'all') and not small_size_test:
            dM_sat_col = Column(data=np.array(dM_sat),name='dM_sat')
            n_sat_col = Column(data=np.array(n_sat),name='n_sat')
            cat_isolated_z_slice_torun.add_column(dM_sat_col)
            cat_isolated_z_slice_torun.add_column(n_sat_col)
            cat_isolated_z_slice_torun.write(scratch_path + cat_name + '_cen_' + str(masscut_host) + '_' + str(round(z_min, 1))+'_'+str(rank_in_cat)+'.fits', overwrite=True)

# combine bootstrap results and calculate error
smf_dist_arr = smf_dist_arr[1:]
smf_dist_bkg_arr = smf_dist_bkg_arr[1:]
smf_dist_sat_arr = smf_dist_arr - smf_dist_bkg_arr

smf_dist_avg = np.average(smf_dist_arr, axis=0)
smf_dist_bkg_avg = np.average(smf_dist_bkg_arr, axis=0)
smf_dist_sat_avg = np.average(smf_dist_sat_arr, axis=0)

smf_dist_inf = np.percentile(smf_dist_arr, 16, axis=0)
smf_dist_bkg_inf = np.percentile(smf_dist_bkg_arr, 16, axis=0)
smf_dist_sat_inf = np.percentile(smf_dist_sat_arr, 16, axis=0)

smf_dist_sup = np.percentile(smf_dist_arr, 84, axis=0)
smf_dist_bkg_sup = np.percentile(smf_dist_bkg_arr, 84, axis=0)
smf_dist_sat_sup = np.percentile(smf_dist_sat_arr, 84, axis=0)

smf_dist_avg_error = [smf_dist_avg, smf_dist_inf, smf_dist_sup]
smf_dist_bkg_avg_error = [smf_dist_bkg_avg, smf_dist_bkg_inf, smf_dist_bkg_sup]
smf_dist_sat_avg_error = [smf_dist_sat_avg, smf_dist_sat_inf, smf_dist_sat_sup]

# output result to file
if len(mass_centrals) == isolated_counts and save_results:
    if (not rel_riso) and (not rel_mscale):
        filename = scratch_path + 'smf_' + cat_name + '_cen_' + str(masscut_host) + '_' + str(r_low) + '_' + str(r_high) + '_' + str(masscut_low) + '_' + csfq + '_' + ssfq + '_' + str(round(z_min, 1))
    elif rel_riso and rel_mscale:
        filename = scratch_path + 'smf_' + cat_name + '_cen_' + str(masscut_host) + '_rel_mscale_' + str(r_low) + '_r200_' + str(masscut_low) + '_' + csfq + '_' + ssfq + '_' + str(round(z_min, 1))
        
    elif rel_mscale:
        filename = scratch_path + 'smf_' + cat_name + '_cen_' + str(masscut_host) + '_rel_mscale_' + str(r_low) +  str(r_high) + '_' + str(masscut_low) + '_' + csfq + '_' + ssfq + '_' + str(round(z_min, 1))
    elif rel_riso:
        if r_factor != 1.0:
            r_factor_infilename = str(r_factor)
        else:
            r_factor_infilename = ''
        filename = scratch_path + 'smf_' + cat_name + '_cen_' + str(masscut_host) + '_' + str(r_low) + '_'+r_factor_infilename+'r200_' + str(masscut_low) + '_' + csfq + '_' + ssfq + '_' + str(round(z_min, 1))
        
    print(filename)
    print(rank_in_cat,'isolated massive counts:', isolated_counts)
    print(rank_in_cat,'avg number total', ssfq, round(sum(smf_dist_avg)/isolated_counts,1))
    print(rank_in_cat,'avg number in bkg', ssfq, round(sum(smf_dist_bkg_avg)/isolated_counts,1))
    print(rank_in_cat,'avg number in sat', ssfq, round(sum(smf_dist_sat_avg)/isolated_counts,1))

    # save total smf (not corrected for bkg)
    smf_dist_avg = np.append(smf_dist_avg, smf_dist_inf)
    smf_dist_avg = np.append(smf_dist_avg, smf_dist_sup)
    smf_dist_avg = np.append([isolated_counts], smf_dist_avg)
    np.save(filename + '_total_'+str(rank_in_cat), smf_dist_avg)

    # save bkg smf
    smf_dist_bkg_avg = np.append(smf_dist_bkg_avg, smf_dist_bkg_inf)
    smf_dist_bkg_avg = np.append(smf_dist_bkg_avg, smf_dist_bkg_sup)
    smf_dist_bkg_avg = np.append([isolated_counts], smf_dist_bkg_avg)
    np.save(filename + '_bkg_'+str(rank_in_cat), smf_dist_bkg_avg)

    # save sat smf
    smf_dist_sat_avg = np.append(smf_dist_sat_avg, smf_dist_sat_inf)
    smf_dist_sat_avg = np.append(smf_dist_sat_avg, smf_dist_sat_sup)
    smf_dist_sat_avg = np.append([isolated_counts], smf_dist_sat_avg)
    np.save(filename + '_sat_'+str(rank_in_cat), smf_dist_sat_avg)

    if not rel_mscale:
        np.save(scratch_path + 'bin_edges', bin_edges)
    else:
        np.save(scratch_path + 'rel_bin_edges', rel_bin_edges)

    
elif not save_results:
    print('isolated counts', isolated_counts, 'bkg counts', count_bkg)
else:
    print(len(mass_centrals), max(mass_centrals), isolated_counts, 'Warning: wrong numbers! (Results not saved)')

