#!/usr/bin/python

#from IPython.core.debugger import Tracer; debug_here = Tracer()

import os
import sys
import shutil
import subprocess
import numpy as np
import nibabel as nib

rootdir = '/media/data/oasis_aal'

rootdir = os.path.abspath(rootdir)

#----------------------------------------------------------------
def exec_comm (comm_line):
    p = subprocess.Popen(comm_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return result

#-------------------------------------------------------------------------------
def find_name_sh (regex, wd='.', args=[]):
    olddir = os.getcwd()
    os.chdir(wd)

    comm = ['find', '-name', regex]
    if args:
        comm.extend(args)

    lst = exec_comm(comm)

    os.chdir(olddir)
    return lst

#-------------------------------------------------------------------------------

outdirnom  = 'aal_rois'
aalvolf    = 'aal_MNI_1mm_FSL.nii.gz'
aal_roisf  = 'aal_allvalues.txt'

os.chdir(rootdir)

roilistf = rootdir + os.path.sep + aal_roisf

#create dictionary for rois: voxel_value -> roi_id
roilist = np.loadtxt(roilistf, dtype=str)
rois = {}
for i in np.arange(roilist.shape[0]):
    rois[int(roilist[i][3])] = roilist[i][0]

#find all aal.to.nodif nifti files
filelist = ['aal_MNI_1mm_FSL.nii.gz']

for fnom in filelist:
    print('Processing ' + fnom)

    #get subject directory
    dirnom = os.path.dirname(fnom)

    #read aal.to.nodif volume
    img = nib.load(fnom)
    vol = img.get_data()
    aff = img.get_affine()

    #create roi directory inside subject's
    outdir = os.path.abspath(dirnom) + os.path.sep + outdirnom

    if not os.path.exists(outdir):
       os.mkdir(outdir)

    #copy aal.to.nodif to outdir
    #shutil.copy(fnom, outdir)

    #cd outdir
    os.chdir(outdir)

    #process volume
    #1. get unique values
    unis = np.unique(vol)
    # delete value 0 form unis

    unis = unis[unis > 0]

    for u in unis:
       #2. create roi mask
       roi = (vol == u) * 1
       roi = roi.copy().astype(np.int16)

       #3. get roi file name
       onom = 'aal_' + rois[u] + '.nii.gz'
       onom = outdir + os.path.sep + onom

       #4. save roi nifti
       ni = nib.Nifti1Image(roi, aff)
       nib.save(ni, onom)
       print('Created ' + onom)

    print('\n')

    os.chdir(rootdir)
