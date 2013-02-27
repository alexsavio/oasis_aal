#!/usr/bin/python

import os
import re
import numpy as np
import nibabel as nib
import scipy.stats as stats

from IPython.core.debugger import Tracer; debug_here = Tracer()

#-------------------------------------------------------------------------------
def get_aal_info(aal_data, roi_idx):
   return aal_data[aal_data[:,3] == str(roi_idx)].flatten()

#-------------------------------------------------------------------------------
def list_filter (list, filter):
    return [ (l) for l in list if filter(l) ]

#-------------------------------------------------------------------------------
def dir_search (regex, wd='.'):
    ls = os.listdir(wd)

    filt = re.compile(regex).search
    return list_filter(ls, filt)

#-------------------------------------------------------------------------------
def dir_match (regex, wd='.'):
    ls = os.listdir(wd)

    filt = re.compile(regex).match
    return list_filter(ls, filt)

#-------------------------------------------------------------------------------
def list_match (regex, list):
    filt = re.compile(regex).match
    return list_filter(list, filt)

#-------------------------------------------------------------------------------
def list_search (regex, list):
    filt = re.compile(regex).search
    return list_filter(list, filt)

#-------------------------------------------------------------------------------
#feats = 'jacs'
#feats = 'smoothmodgm'
#feats = 'geodan'
#feats = 'norms'
feats = 'trace'

#ftype = 'raw'
ftype  = 'stats'
nfeats = 7


if feats == 'jacs' or feats == 'geodan' or feats == 'norms' or feats == 'trace':
    rootdir  = '/media/data/oasis_aal'
    subjsdir = '/data/oasis_jesper_features/' + feats

    subjlst = dir_match(r"(.)+.nii.gz", subjsdir)

elif feats == 'smoothmodgm':
    rootdir  = '/media/data/oasis_aal'
    subjsdir = '/media/data/oasis_aal/oasis_nn'

    subjlst = dir_match(r"(.)+_smooth(.)+", subjsdir)

outdir      = rootdir + os.path.sep + 'oasis_' + feats + '_feats'
outbasename = 'oasis_' + feats + '_' + ftype
roisdir     = rootdir + os.path.sep + 'aal_rois'
aalf        = rootdir + os.path.sep + 'aal_allvalues.txt'

subjlst.sort()
nsubjs  = len(subjlst)

#create outdir if it does not exist
if not os.path.exists(outdir):
    os.mkdir(outdir)

#get info from ROIs
aalinfo = np.loadtxt (aalf, dtype=str)
roilst  = aalinfo[:,0]
nrois  = len(roilst)

#get a list of the aal roi volumes
roifs  = dir_search('aal.smooth*', roisdir)

#save list of subjects
subjlsf = outdir + os.path.sep + outbasename + '_subjlist.txt'
np.savetxt(subjlsf, subjlst, fmt='%s')

#create space for all features and read from subjects
for r in roilst:
    roi    = r
    roif   = list_search(roi, roifs)[0]
    aalidx = [i for i, x in enumerate(aalinfo[:,0]) if x == roi]
    aalrow = aalinfo[aalidx,:]

    #load roi
    roivol = nib.load(roisdir + os.path.sep + roif).get_data()

    if ftype == 'raw':
        nfeats = np.sum  (roivol > 0)
        feats  = np.zeros((nsubjs, nfeats))
    elif ftype == 'stats':
        feats  = np.zeros((nsubjs, 7))

    print ('Processing ' + roi)
    for s in np.arange(nsubjs):
        subjf = subjsdir + os.path.sep + subjlst[s]
        subj  = nib.load(subjf).get_data()

        fs = subj[roivol > 0]

        if ftype == 'raw':
            feats[s,:] = fs

        elif ftype == 'stats':
            feats[s,0] = np.max         (fs)
            feats[s,1] = np.min         (fs)
            feats[s,2] = np.mean        (fs)
            feats[s,3] = np.var         (fs)
            feats[s,4] = np.median      (fs)
            feats[s,5] = stats.kurtosis (fs)
            feats[s,6] = stats.skew     (fs)

    #save file
    outfname = outdir + os.path.sep + outbasename + '_' + roi + '.npy'
    print ('Saving ' + outfname)
    np.save(outfname, feats)

