from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from astropy.stats import bootstrap
from astropy.cosmology import WMAP9
import numpy as np
import sys
from mpi4py import MPI
comm = MPI.COMM_WORLD
n_procs = comm.Get_size()
rank = comm.Get_rank()

bin_number = 6
all_gals = np.zeros(bin_number)
sf_gals =np.zeros(bin_number)
q_gals = np.zeros(bin_number)
detected_gals = np.zeros(bin_number)
detected_sfs = np.zeros(bin_number)
detected_qs = np.zeros(bin_number)

bin_type = sys.argv[1]
if bin_type == 'mass':
    masscut_low = float(sys.argv[2])
    masscut_high = float(sys.argv[3])
else:
    magcut_low = float(sys.argv[2])
    magcut_high = float(sys.argv[3])

z=eval(sys.argv[4])
z_low = round(float(sys.argv[4])-0.1, 1)
z_high = round(float(sys.argv[4])+0.1, 1)
sfq_keyname = sys.argv[5]
sat_z_cut = float(sys.argv[6])

cat_names = ['COSMOS_deep', 'DEEP_deep', 'XMM-LSS_deep', 'ELAIS_deep']
csfq = 'all'

# parent process: collect all results
if rank == n_procs-1:
    for i in range(n_procs-1):
        all_gal, sf_gal, q_gal,detected_gal,detected_sf,detected_q,all_curve, sf_curve, q_curve = comm.recv(source=MPI.ANY_SOURCE)
        if i == 0:
            all_gals = all_gal
            sf_gals = sf_gal
            q_gals = q_gal
            detected_gals = detected_gal
            detected_sfs = detected_sf
            detected_qs = detected_q
            all_curves = all_curve
            sf_curves = sf_curve
            q_curves = q_curve
        else:
            all_gals = np.vstack((all_gals, all_gal))
            sf_gals = np.vstack((sf_gals, sf_gal))
            q_gals = np.vstack((q_gals, q_gal))
            detected_gals = np.vstack((detected_gals, detected_gal))
            detected_sfs = np.vstack((detected_sfs, detected_sf))
            detected_qs = np.vstack((detected_qs, detected_q))
            all_curves = np.vstack((all_curves, all_curve))
            sf_curves = np.vstack((sf_curves, sf_curve))
            q_curves = np.vstack((q_curves, q_curve))

    path_curve = 'curves_700/'
    path_stats = 'stats/'
    if bin_type == 'mag':
        np.savetxt(path_curve + 'comp_'+ bin_type +'_' + str(sat_z_cut)+'_'+csfq+'_all_'+sfq_keyname+'_'+str(magcut_low)+'_'+str(magcut_high)+'_'+str(z_low)+'_'+str(z_high)+'.txt', all_curves)
        np.savetxt(path_curve + 'comp_'+ bin_type +'_'+str(sat_z_cut)+'_'+csfq+'_ssf_'+sfq_keyname+'_'+str(magcut_low)+'_'+str(magcut_high)+'_'+str(z_low)+'_'+str(z_high)+ '.txt', sf_curves)
        np.savetxt(path_curve + 'comp_'+ bin_type +'_'+str(sat_z_cut)+'_'+csfq+'_sq_' +sfq_keyname+'_'+str(magcut_low)+'_'+str(magcut_high)+'_'+str(z_low)+'_'+str(z_high)+'.txt', q_curves)

        np.savetxt(path_stats + 'total_'+ bin_type +'_' + str(sat_z_cut) + '_' + csfq + '_all_' + sfq_keyname + '_' + str(magcut_low) + '_' + str(magcut_high) + '_' + str(z_low) + '_' + str(z_high) + '.txt', all_gals)
        np.savetxt(path_stats + 'total_'+ bin_type +'_' + str(sat_z_cut) + '_' + csfq + '_ssf_' + sfq_keyname + '_' + str(magcut_low) + '_' + str(magcut_high) + '_' + str(z_low) + '_' + str(z_high) + '.txt', sf_gals)
        np.savetxt(path_stats + 'total_'+ bin_type +'_' + str(sat_z_cut) + '_' + csfq + '_sq_' + sfq_keyname + '_' + str(magcut_low) + '_' + str( magcut_high) + '_' + str(z_low) + '_' + str(z_high) + '.txt', q_gals)
        np.savetxt(path_stats + 'redetect_'+ bin_type +'_' + str(sat_z_cut) + '_' + csfq + '_all_' + sfq_keyname + '_' + str(magcut_low) + '_' + str(magcut_high) + '_' + str(z_low) + '_' + str(z_high) + '.txt', detected_gals)
        np.savetxt(path_stats + 'redetect_'+ bin_type +'_' + str(sat_z_cut) + '_' + csfq + '_ssf_' + sfq_keyname + '_' + str(magcut_low) + '_' + str(magcut_high) + '_' + str(z_low) + '_' + str(z_high) + '.txt', detected_sfs)
        np.savetxt(path_stats + 'redetect_'+ bin_type +'_' + str(sat_z_cut) + '_' + csfq + '_sq_' + sfq_keyname + '_' + str(magcut_low) + '_' + str(magcut_high) + '_' + str(z_low) + '_' + str(z_high) + '.txt', detected_qs)

    else:
        np.savetxt(path_curve + 'comp_' + str(sat_z_cut) + '_' + csfq + '_all_' + sfq_keyname + '_' + str(masscut_low) + '_' + str(masscut_high) + '_' + str(z_low) + '_' + str(z_high) + '.txt', all_curves)
        np.savetxt(path_curve + 'comp_' + str(sat_z_cut) + '_' + csfq + '_ssf_' + sfq_keyname + '_' + str(masscut_low) + '_' + str(masscut_high) + '_' + str(z_low) + '_' + str(z_high) + '.txt', sf_curves)
        np.savetxt(path_curve + 'comp_' + str(sat_z_cut) + '_' + csfq + '_sq_' + sfq_keyname + '_' + str(masscut_low) + '_' + str(masscut_high) + '_' + str(z_low) + '_' + str(z_high) + '.txt', q_curves)

        np.savetxt(path_stats + 'total_' + str(sat_z_cut) + '_' + csfq + '_all_' + sfq_keyname + '_' + str( masscut_low) + '_' + str(masscut_high) + '_' + str(z_low) + '_' + str(z_high) + '.txt', all_gals)
        np.savetxt(path_stats + 'total_' + str(sat_z_cut) + '_' + csfq + '_ssf_' + sfq_keyname + '_' + str(masscut_low) + '_' + str(masscut_high) + '_' + str(z_low) + '_' + str(z_high) + '.txt', sf_gals)
        np.savetxt(path_stats + 'total_' + str(sat_z_cut) + '_' + csfq + '_sq_' + sfq_keyname + '_' + str(masscut_low) + '_' + str(masscut_high) + '_' + str(z_low) + '_' + str(z_high) + '.txt', q_gals)
        np.savetxt(path_stats + 'redetect_' + str(sat_z_cut) + '_' + csfq + '_all_' + sfq_keyname + '_' + str(masscut_low) + '_' + str(masscut_high) + '_' + str(z_low) + '_' + str(z_high) + '.txt', detected_gals)
        np.savetxt(path_stats + 'redetect_' + str(sat_z_cut) + '_' + csfq + '_ssf_' + sfq_keyname + '_' + str(masscut_low) + '_' + str(masscut_high) + '_' + str(z_low) + '_' + str(z_high) + '.txt', detected_sfs)
        np.savetxt(path_stats + 'redetect_' + str(sat_z_cut) + '_' + csfq + '_sq_' + sfq_keyname + '_' + str(masscut_low) + '_' + str(masscut_high) + '_' + str(z_low) + '_' + str(z_high) + '.txt', detected_qs)

# children processes (I said the calculation!)
else:
    bin_edges = 10 ** np.linspace(90, np.log10(700), num=bin_number + 1)
    for cat_name in cat_names:
        print('==============='+cat_name+'==================')
        print('z range', z_low, z_high)

        # loading /cleaning of mock catalpg
        mock_cat = Table.read('Output_cats/phys_matched_new_cat_stack_'+cat_name+'_'+str(z)+'.fits')
        mock_cat = mock_cat[~np.isnan(np.array(mock_cat['MASS_MED']))]  # clean the catalog for mass measurement
        mock_cat = mock_cat[~np.isnan(np.array(mock_cat['ZPHOT']))]
        mock_cat = mock_cat[~np.isnan(np.array(mock_cat['i']))]
        mock_cat = mock_cat[~np.isnan(np.array(mock_cat['RA']))]
        mock_cat = mock_cat[~np.isnan(np.array(mock_cat['X_WORLD']))]
        mock_cat = mock_cat[~np.isnan(np.array(mock_cat[sfq_keyname]))]
        mock_cat = mock_cat[mock_cat['ORIGINAL'] == False]  # only keep mock objects in mock cat
        if bin_type == 'mass':
            mock_cat = mock_cat[mock_cat['MASS_MED'] > masscut_low]  # mass cut
            mock_cat = mock_cat[mock_cat['MASS_MED'] < masscut_high]  # mass cut
            print('After mass cut, len==' + str(len(mock_cat)))
        else:
            mock_cat = mock_cat[mock_cat['i'] > magcut_low]  # mass cut
            mock_cat = mock_cat[mock_cat['i'] < magcut_high]  # mass cut
            print('After mag cut, len==' + str(len(mock_cat)))

        # loading / cleaning of central catalog
        cat_massive_gal = Table.read('/home/lejay/radial_dist_code/central_cat/isolated_' + cat_name + '_11.15_' + str(z) + '_massive.positions.fits')
        cat_massive_gal = cat_massive_gal[~np.isnan(cat_massive_gal['ZPHOT'])]
        cat_massive_gal = cat_massive_gal[~np.isnan(cat_massive_gal['MASS_MED'])]
        cat_massive_gal = cat_massive_gal[cat_massive_gal['ZPHOT'] > z_low]  # redshift cut for massive galaxies
        cat_massive_gal = cat_massive_gal[cat_massive_gal['ZPHOT'] < z_high]  # redshift cut for massive galaxies

        # gal ids
        gal_ids = open('Gal_ids/gal_ids_'+cat_name+'_'+str(z)+'.txt').readlines()
        for i in range(len(gal_ids)):
            gal_ids[i] = gal_ids[i].rstrip()
        print('No. of centrals:', len(gal_ids))

        no_sat_gals = 0
        for gal in cat_massive_gal:
            if str(gal['ID']) in gal_ids:
                dis = WMAP9.angular_diameter_distance(float(gal['ZPHOT'])).value

                # selection for potential satellites (in mock catalog)
                cat_neighbors = mock_cat[abs(mock_cat['X_WORLD'] - gal['RA']) < 2.5 / dis / np.pi * 180]
                cat_neighbors = cat_neighbors[abs(cat_neighbors['Y_WORLD'] - gal['DEC']) < 2.5 / dis / np.pi * 180]
                cat_neighbors = cat_neighbors[abs(cat_neighbors['ZPHOT'] - gal['ZPHOT']) < sat_z_cut*0.044*(1+gal['ZPHOT'])]

                # cleaning for the sfq key
                if 'sfq' in sfq_keyname:
                    cat_neighbors = cat_neighbors[np.logical_or(cat_neighbors[sfq_keyname]==0,cat_neighbors[sfq_keyname]==1)]
                elif 'sfProb' in sfq_keyname:
                    cat_neighbors = cat_neighbors[np.logical_and(cat_neighbors[sfq_keyname]>0, cat_neighbors[sfq_keyname]<1)]

                # bootstrap
                if len(cat_neighbors) > 0:
                    boot_idx = bootstrap(np.arange(len(cat_neighbors)), bootnum=1)
                    cat_neighbors = cat_neighbors[boot_idx[0].astype(int)]
                else:
                    no_sat_gals += 1
                    continue

                # prepare radius list
                coord_gal = SkyCoord(gal['RA'] * u.deg, gal['DEC'] * u.deg)
                coord_neighbors = SkyCoord(cat_neighbors['X_WORLD'] * u.deg, cat_neighbors['Y_WORLD'] * u.deg)
                radius_list = coord_neighbors.separation(coord_gal).degree / 180. * np.pi * dis * 1000  # kpc

                # count of total number
                all = np.histogram(radius_list, bins=bin_edges)[0]
                if 'sfProb' in sfq_keyname:
                    sf = np.histogram(radius_list, weights=cat_neighbors[sfq_keyname], bins=bin_edges)[0]
                    q = np.histogram(radius_list, weights=1-cat_neighbors[sfq_keyname], bins=bin_edges)[0]
                elif 'sfq' in sfq_keyname:
                    sf = np.histogram(radius_list[cat_neighbors[sfq_keyname]==0], bins=bin_edges)[0]
                    q = np.histogram(radius_list[cat_neighbors[sfq_keyname]==1], bins=bin_edges)[0]
                elif sfq_keyname == 'SSFR_MED':
                    sf = np.histogram(radius_list[cat_neighbors['SSFR_MED'] > -11], bins=bin_edges)[0]
                    q = np.histogram(radius_list[cat_neighbors['SSFR_MED'] < -11], bins=bin_edges)[0]

                # prepare recovered radius list
                cat_detected = cat_neighbors[~np.isnan(cat_neighbors['FLUX_APER_1.0'])]
                coord_detected = SkyCoord(cat_detected['X_WORLD'] * u.deg, cat_detected['Y_WORLD'] * u.deg)
                radius_list_detected = coord_detected.separation(coord_gal).degree / 180. * np.pi * dis * 1000  # kpc

                # recovered
                detected = np.histogram(radius_list_detected, bins=bin_edges)[0]
                if 'sfProb' in sfq_keyname:
                    detected_sf = np.histogram(radius_list_detected, weights=cat_detected[sfq_keyname], bins=bin_edges)[0]
                    detected_q = np.histogram(radius_list_detected, weights=1-cat_detected[sfq_keyname], bins=bin_edges)[0]
                elif 'sfq' in sfq_keyname:
                    detected_sf = np.histogram(radius_list_detected[cat_detected[sfq_keyname]==0], bins=bin_edges)[0]
                    detected_q = np.histogram(radius_list_detected[cat_detected[sfq_keyname]==1], bins=bin_edges)[0]
                elif sfq_keyname == 'SSFR_MED':
                    detected_sf = np.histogram(radius_list_detected[cat_detected['SSFR_MED'] > -11], bins=bin_edges)[0]
                    detected_q = np.histogram(radius_list_detected[cat_detected['SSFR_MED'] < -11], bins=bin_edges)[0]

                all_gals += all
                sf_gals += sf
                q_gals += q
                detected_gals += detected
                detected_sfs += detected_sf
                detected_qs += detected_q

    all_gals[all_gals==0] = 1
    sf_gals[sf_gals==0] = 1
    q_gals[q_gals==0] = 1
    all_curve = detected_gals / all_gals
    sf_curve = detected_sfs / sf_gals
    q_curve = detected_qs / q_gals

    print('no sat gals', no_sat_gals)
    print('detected all', sum(detected_gals), detected_gals)
    print('detected sf', sum(detected_sfs), detected_sfs)
    print('detected q', sum(detected_qs), detected_qs)
    print('all',sum(all_gals), all_gals)
    print('sf',sum(sf_gals), sf_gals)
    print('q',sum(q_gals), q_gals)

    np.set_printoptions(precision=3)
    print('all', all_curve)
    print('sf', sf_curve)
    print('q', q_curve)

    comm.send((all_gals,sf_gals,q_gals,detected_gals,detected_sfs,detected_qs,all_curve, sf_curve, q_curve), dest=n_procs-1)
    np.save('bin_edges.npy', bin_edges)
