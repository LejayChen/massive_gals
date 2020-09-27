from astropy.table import Table
import sys, os

run_name = sys.argv[1]
cat_name = sys.argv[2]

# read in pipes catalog
pipes_cat = Table.read('pipes/cats/'+run_name+'.fits')
pipes_cat_cut = pipes_cat
# try:
#     pipes_cat_cut = pipes_cat['#ID', 'stellar_mass_16', 'stellar_mass_50', 'stellar_mass_84',
#                           'delayed:massformed_16', 'delayed:massformed_50', 'delayed:massformed_84',
#                           'formed_mass_16', 'formed_mass_50', 'formed_mass_84',
#                           'redshift_16', 'redshift_50', 'redshift_84',
#                           'ssfr_16', 'ssfr_50', 'ssfr_84']
# except KeyError:
#     pipes_cat_cut = pipes_cat['#ID', 'stellar_mass_16', 'stellar_mass_50', 'stellar_mass_84',
#                               'delayed:massformed_16', 'delayed:massformed_50', 'delayed:massformed_84',
#                               'formed_mass_16', 'formed_mass_50', 'formed_mass_84',
#                               'input_redshift',
#                               'ssfr_16', 'ssfr_50', 'ssfr_84']

pipes_cat_cut.rename_column('#ID', 'ID')
pipes_cat_cut['ID'] = pipes_cat_cut['ID'].astype('i8')

# output modified pipes catalog
output_name = cat_name+'_'+run_name+'.fits'
pipes_cat_cut.write(output_name, overwrite=True)

# combine with the original catalog
cmd ='java -jar  -Xms128m -Xmx1024m ../completeness_curve/stilts.jar tmatch2 in1='+output_name+' in2=../catalogs/'+cat_name+'_to_fit_'+sys.argv[3]+'.fits join=1and2 \
                                                      matcher=exact values1="ID" values2="ID" \
                                                      out='+cat_name+'_pipes_added_'+sys.argv[4]+'.fits'
os.system(cmd)
os.system('rm '+output_name)
