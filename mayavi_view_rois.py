#!/usr/bin/python

import os
import sys
import numpy as np
import scipy.ndimage as scn
import pylab as pl
import nibabel as nib
from mayavi import mlab

sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/visualize_volume')
import visualize_volume as vis

#--------------------------------------------------------------------
def load_strdata_file (dir, fname):
    return np.loadtxt(os.path.join(dir, fname), dtype=str)

#--------------------------------------------------------------------
def filter_volume (vol, values):
    if len(values) < 1:
        return vol

    res = np.zeros_like(vol)
    for i in values:
        res[vol == i] = i

    return res

#--------------------------------------------------------------------
def save_nibabel (ofname, vol, affine, header=None):
   #saves nifti file
   ni = nib.Nifti1Image(vol, affine, header)
   nib.save(ni, ofname)
#--------------------------------------------------------------------

wd   = '/home/alexandre/Dropbox/Documents/phd/work/oasis_aal'
aalf = os.path.join(wd, 'aal_MNI_1mm_FSL.nii.gz')
mnif = '/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'

mniv = nib.load(mnif).get_data()
aalv = nib.load(aalf).get_data()

rois   = load_strdata_file(wd, 'aal_allvalues.txt')
c1rois = load_strdata_file(wd, 'aal_allvalues_Alzheimer_CDR1.txt')
mirois = load_strdata_file(wd, 'aal_allvalues_Alzheimer_Mild.txt')
alrois = load_strdata_file(wd, 'aal_allvalues_Alzheimer.txt')

c1roisv = filter_volume (aalv, c1rois[:,3].astype(int))
miroisv = filter_volume (aalv, mirois[:,3].astype(int))
alroisv = filter_volume (aalv, alrois[:,3].astype(int))

#aalh  = nib.load(aalf).get_header()
#aalaf = nib.load(aalf).get_affine()
#save_nibabel (os.path.join(wd, 'aal_MNI_1mm_CDR1.nii.gz'), c1roisv, #aalaf, aalh)
#save_nibabel (os.path.join(wd, 'aal_MNI_1mm_MildAD.nii.gz'), #miroisv, aalaf, aalh)
#save_nibabel (os.path.join(wd, 'aal_MNI_1mm_AD.nii.gz'), alroisv, aalaf, aalh)

mlab.figure(1, fgcolor=(1, 1, 1), bgcolor=(1, 1, 1))
color = 'hsv'

img  = mniv
roiv = miroisv
src = mlab.pipeline.scalar_field(img)

c = 1

src.image_data.point_data.add_array(roiv.T.ravel())

src.image_data.point_data.get_array(0).name = 'scalar'
src.image_data.point_data.get_array(1).name = 'rois'
src.image_data.point_data.update()


blur = mlab.pipeline.user_defined(src, filter='ImageGaussianSmooth')

mlab.pipeline.iso_surface(src, contours=[img.min()+0.1*img.ptp(), ], opacity=0.1)

src2 = mlab.pipeline.set_active_attribute(src,
                                    point_scalars='rois')

mlab.pipeline.volume(src2)

mlab.pipeline.iso_surface(src2, colormap=color)

#src = mlab.pipeline.scalar_field(timg)
#mlab.pipeline.iso_surface(src, contours=[timg.min()+0.1*timg.ptp(), ], opacity=0.4, colormap='gray')

#mlab.pipeline.iso_surface(src1, contours=[timg.max()-0.1*timg.ptp(), ],)


#src2 = mlab.pipeline.scalar_field(dimg)



mlab.savefig('example.png')




mlab.pipeline.scalar_cut_plane(src3,
                            plane_orientation='z_axes',
                            colormap=color,
                            transparent=True,
                            opacity=0.1,
                        )

mlab.pipeline.image_plane_widget(src3,
                            plane_orientation='x_axes',
                            slice_index=10,
                            colormap=color,
                            transparent=True,
                            opacity=0.1,
                        )

src2 = mlab.pipeline.set_active_attribute(src,
                                          point_scalars='scalar')

mlab.pipeline.iso_surface(src2, contours=[timg.min()+0.1*timg.ptp(), ], opacity=0.2, colormap='gray')

#mlab.pipeline.iso_surface(src2, contours=[dimg.min()+0.1*dimg.ptp(), ], opacity=0.6, colormap='hot')

#mlab.pipeline.surface(src, colormap=color)
mlab.colorbar(title='', orientation='vertical', nb_labels=6, label_fmt='%.2f')
