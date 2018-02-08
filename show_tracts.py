from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.lines as lines


#COSMOS_tracts = ['9570', '9571', '9812', '9813', '9814', '10054', '10055']
XMM_tracts = ['8523', '8524', '8765', '8766']


def find_corners(tract_number):
    tract_info = open('tracts_patches/tracts_patches_D-XMM-LSS.txt')
    corners = []
    for line in tract_info.readlines():
        if tract_number in line and len(line.split()) == 8 and 'Center' not in line:
            corners.append([eval(line.split()[-3][1:]), eval(line.split()[-1][0:-1])])
    return corners

fig = plt.figure()
ax = fig.add_subplot(111)
cat = Table.read('CLAUDS_HSC_VISTA_Ks23.3_PHYSPARAM_TM.fits')
ax.plot(cat['RA'],cat['DEC'],'.', color='r', alpha=0.1)

for tract in XMM_tracts:
    corners = find_corners(tract)
    ax.add_line(lines.Line2D((corners[0][0],corners[1][0]), (corners[0][1],corners[1][1])))
    ax.add_line(lines.Line2D((corners[2][0],corners[3][0]), (corners[2][1],corners[3][1])))
    ax.add_line(lines.Line2D((corners[0][0], corners[3][0]), (corners[0][1], corners[3][1])))
    ax.add_line(lines.Line2D((corners[1][0], corners[2][0]), (corners[1][1], corners[2][1])))
    ax.text(corners[0][0]-1,corners[0][1]+0.5,tract)


ax.axis([33, 37, -6.5, -2.5])
ax.set_title('XMM-LSS')
plt.show()