import os
import numpy as np
import nibabel as nib

wd = '/home/alexandre/Dropbox/Documents/phd/work/oasis_aal'

#roilist
roilist = os.listdir(os.path.join(wd,'borja/mat/oasis_geodan_stats'))
roilist = np.array(roilist, dtype=str)

c = 0
for f in roilist:
    roinom = str.split(f.replace("oasis_geodan_stats_", ""), ".")[0]
    roilist[c] = roinom
    c += 1

roilist.sort()
roilist = roilist[0:-1]

#results smoothmodgm indexes
bdcidx = np.array([10, 60, 87, 88, 61, 48, 106, 115 ,37 ,9])
elmidx = np.array([60, 88, 6, 7, 5, 10, 9, 8, 115, 85, 104, 112, 12, 15, 67, 18, 16, 21, 94, 98, 64, 32, 49, 59, 75, 3, 108, 46, 109, 4])
hrfidx = np.array([88, 33, 24, 23, 28, 8, 59, 68, 2, 98, 74, 85, 89, 27, 1, 3, 67, 95, 13])
rfidx  = np.array([69, 29, 88, 50, 30, 59, 62, 74, 104, 68, 70, 113, 112, 9, 36, 38, 60, 97, 116, 71, 72, 75, 16, 49, 86, 102, 110, 11, 18, 96, 61, 103, 32, 85, 28, 54, 21, 80, 93, 7, 15, 48, 115, 44, 67, 94, 39, 90, 31])

#the above indexes are defined from 1 to 116, instead python counts from 0 to 115:
bdcidx = bdcidx - 1
elmidx = elmidx - 1
hrfidx = hrfidx - 1
rfidx  =  rfidx - 1

#get roi names
bdcrois = roilist[bdcidx]
elmrois = roilist[elmidx]
hrfrois = roilist[hrfidx]
rfrois  = roilist[ rfidx]

#nifti
def save_nibabel (ofname, vol, affine, header=None):
   #saves nifti file
   ni = nib.Nifti1Image(vol, affine, header)
   nib.save(ni, ofname)

#aal volume
aalf  = os.path.join(wd, 'aal_MNI_1mm_FSL.nii.gz')
aalv  = nib.load(aalf).get_data  ()
aalh  = nib.load(aalf).get_header()
aalaf = nib.load(aalf).get_affine()

#aal roi values
roisf = os.path.join(wd, 'aal_allvalues.txt')
aalinfo = np.loadtxt(roisf, dtype=str)
aalvals = {}
aalnoms = {}
for i in np.arange(len(aalinfo)):
    aalvals[aalinfo[i,0]] = int(aalinfo[i,3])
    aalnoms[aalinfo[i,0]] = aalinfo[i,1]

#save results
resrois = {}
resrois['bdc'] = bdcrois
resrois['elm'] = elmrois
resrois['hrf'] = hrfrois
resrois['rf']  = rfrois

for r in resrois.keys():
    rrois = resrois[r]
    print r
    for n in rrois:
        print n, aalnoms[n]
    print

    rvol = np.zeros_like(aalv)
    for n in rrois:
        rvol[aalv == aalvals[n]] = aalvals[n]

    of = os.path.join(wd, 'aal_borja_' + r + '.nii.gz')
    print ('saving ' + of)
    save_nibabel (of, rvol, aalaf, aalh)


