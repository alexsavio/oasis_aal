#!/usr/bin/python

import os
import re
import sys
import argparse
import numpy as np
import nibabel as nib

import scipy.stats as stats

from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from IPython.core.debugger import Tracer; debug_here = Tracer()

sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/aizkolari')
import aizkolari_utils as au


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
def prepare_asvm_args (trainfname, testfname):
    parser = asvm.set_parser()

    parser.set_defaults(method = a)
    parser.add_argument('-f', type = str)
    parser.add_argument('', type = str)

    arguments = parser.parse_args()
    arguments.method(**vars(arguments))

    return arguments

#-------------------------------------------------------------------------------
#class my_class_metrics:
#    def __init__(self, accuracy=0.0, recall=0.0, precision=0.0, specificity=0.0, roc_auc=0.0):
#        self.accuracy    = accuracy
#        self.recall      = recall
#        self.precision   = precision
#        self.specificity = specificity
#        self.roc_auc     = roc_auc

#-------------------------------------------------------------------------------
def all_classif_metrics (cv_targets, cv_preds, cv_probs=None):

    if (isinstance(cv_targets, dict)):
        metrics = np.zeros((len(cv_targets.keys()), 5))
        rango   = cv_targets.keys()

        c = 0
        for i in rango:
            try:
                targets = cv_targets[i]
                preds   = cv_preds  [i]
                probs   = cv_probs  [i]
            except:
                print( "Unexpected error: ", sys.exc_info()[0] )
                debug_here()

            acc, sens, spec, f1, fpr, tpr, auc = classification_metrics (targets, preds, probs)
            metrics[c, :] = np.array([acc, sens, spec, f1, auc])
            c += 1

    else:
        metrics = np.zeros((cv_targets.shape[0], 6))
        rango   = np.arange(cv_targets.shape[0])

        for i in rango:
            targsi = cv_targets[i,:]
            predsi = cv_preds  [i,:]

            if cv_probs != None: probsi = cv_probs[i,:,:]
            else: probsi = None

            acc, sens, spec, prec, f1, fpr, tpr, auc = classification_metrics (targsi, predsi, probsi)

            metrics[i, :] = np.array([acc, sens, spec, prec, f1, auc])

    return metrics

#-------------------------------------------------------------------------------
def classification_metrics (targets, preds, probs=None):

    if probs != None:
        fpr, tpr, thresholds = roc_curve(targets, probs[:, 1], 1)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, thresholds = roc_curve(targets, preds, 1)
        roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(targets, preds)

    #accuracy
    acc = accuracy_score(targets, preds)

    #recall? True Positive Rate or Sensitivity or Recall
    sens = recall_score(targets, preds)

    #precision
    prec = precision_score(targets, preds)

    #f1-score
    f1 = f1_score(targets, preds, np.unique(targets), 1)

    tnr = 0.0
    #True Negative Rate or Specificity (tn / (tn+fp))
    if len(cm) == 2:
        spec = float(cm[0,0])/(cm[0,0] + cm[0,1])

    return acc, sens, spec, prec, f1, fpr, tpr, roc_auc

#-------------------------------------------------------------------------------
def print_pred_metrics (truth, pred):

    print ('Accuracy - Sensitivity - Specificity - Precision - F1-score - AUC')

    if pred.ndim > 1: 
        n_preds = preds.shape[1]
        for i in np.arange(n_preds):
            acc, sens, spec, prec, f1, fpr, tpr, roc_auc = classification_metrics(truth, pred[:,i])
            print ('%.4f - %.4f - %.4f - %.4f - %.4f - %.4f' % (acc, sens, spec, prec, f1, roc_auc))

    else:
        acc, sens, spec, prec, f1, fpr, tpr, roc_auc = classification_metrics(truth, pred)

        print ('%.4f - %.4f - %.4f - %.4f - %.4f - %.4f' % (acc, sens, spec, prec, f1, roc_auc))


#-------------------------------------------------------------------------------
def plot_roc_curve (targets, preds, probs):
    import pylab as pl
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(targets, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    pl.plot(fpr, tpr, lw=1, label='ROC LOO-test (area = %0.2f)' % (roc_auc))
    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('ROC for ' + feats + ' ROI ' + roinom)
    pl.legend(loc="lower right")
    pl.show()

#-------------------------------------------------------------------------------

def majority (preds):
    unis    = np.unique(preds)
    maj = 0
    sm  = 0
    for u in unis:
        cu = np.sum(preds == u)
        if cu > sm:
            sm  = cu
            maj = u
    return maj

#-------------------------------------------------------------------------------
def common_prefix(strings):
    """ Find the longest string that is a prefix of all the strings.
    """
    if not strings:
        return ''
    prefix = strings[0]
    for s in strings:
        if len(s) < len(prefix):
            prefix = prefix[:len(s)]
        if not prefix:
            return ''
        for i in range(len(prefix)):
            if prefix[i] != s[i]:
                prefix = prefix[:i]
                break
    return prefix

#-------------------------------------------------------------------------------
def remove_common_prefix (strings, avoid):
    cp = common_prefix (strings)
    
    if len(cp) > 0:
        return [ (l.replace(cp, '')) for l in strings ]

    return strings

#-------------------------------------------------------------------------------
def find_substr (substr, lst):
    res = list_match (substr, lst)
    if len(res) == 1:
        return res[0]
    elif len(res) > 1:
        debug_here()

    res = list_search (substr, lst)
    if len(res) == 1:
        return res[0]
    else:
        subres = remove_common_prefix (res)
        idx    = np.where(np.array(subres) == substr)
        if len(idx[0]) > 0:
            return res[idx[0][0]]
        else:
            debug_here()

#-------------------------------------------------------------------------------
def get_predictions (results, rois, n_subjs):

    n_rois = len(rois)

    truth  = np.zeros((n_rois, n_subjs), dtype=int)
    preds  = np.zeros_like(truth)
    probs  = np.zeros((n_rois, n_subjs, 2))
    taucs  = np.zeros((n_rois, n_subjs)) #training AUC
    tfones = np.zeros((n_rois, n_subjs)) #training F1-scores

    c = 0
    for r in rois:

        k = find_substr(r, res.keys())

        truth [c,:] = np.concatenate(res[k]['truth'].values())
        preds [c,:] = np.concatenate(res[k]['preds'].values())
        taucs [c,:] = res[k]['train_auc_scores'].values()
        tfones[c,:] = res[k]['train_f1_scores' ].values()

        if (len(res[k]['probs']) > 0):
            probs[c,:,:] = np.concatenate(res[k]['probs'].values())
        c += 1

    return truth, preds, probs, taucs, tfones

#-------------------------------------------------------------------------------
def get_rois_indexes (roisf, reflst):
    nurois = np.loadtxt(roisf, dtype=str)[:,0]
    idx    = np.zeros(len(nurois), dtype=int)
    c = 0
    for r in nurois:
        idx[c] = np.where(reflst == r)[0][0]
        c += 1
    return idx

#-------------------------------------------------------------------------------

hn = au.get_hostname()
if hn == 'azteca':
    roisf='/home/alexandre/Dropbox/Documents/phd/work/oasis_aal/aal_allvalues.txt'
    wd = '/home/alexandre/Dropbox/Documents/phd/work/oasis_aal'
elif hn == 'corsair':
    roisf='/home/alexandre/Dropbox/Documents/phd/work/oasis_aal/aal_allvalues.txt'
    wd = '/home/alexandre/Dropbox/Documents/phd/work/oasis_aal'
elif hn == 'hpmed':
    roisf='/home/alexandre/Dropbox/Documents/phd/work/oasis_aal/aal_allvalues.txt'
    wd = '/home/alexandre/Dropbox/Documents/phd/work/oasis_aal'

roilst = np.loadtxt(roisf, dtype=str)
rois   = roilst[:,0]
n_rois = len(rois)

#cvfolds = '10'
cvfolds = 'loo'

piklst = dir_match("(.)+_" + cvfolds + "_(.)+.pickle", wd)
piklst.sort()

f = piklst[14]
for f in piklst:
    print ('***********************************************')
    print ('-----------------------------------------------')
    print (f)

    res = np.load(os.path.join(wd, f))

    fsfiles = res['fsfiles']
    y       = res['y']
    subjs   = res['subjs']
    n_subjs = len(y)

    if (n_rois + 3) != len(res.keys()):
        del res['fsfiles']
        del res['y']
        del res['subjs']
        rois   = np.sort(res.keys())
        n_rois = len(rois)

    #populate ROI metrics and save all predictions
    if cvfolds == 'loo':
        truth, preds, probs, taucs, tfones = get_predictions (res, rois, n_subjs)
        metrics = all_classif_metrics (truth, preds, probs)

    elif cvfolds == '10':
        metrics = {}
        for r in rois:
            metrics[r] = all_classif_metrics_dicts (res[r]['truth'], res[r]['preds'], res[r]['probs'])

    #calculate metrics for each ROI separately
    if cvfolds == '10':
        print ('-----------------------------------------------')
        print 'Each ROI separately'

        i = 0
        mean_met = np.zeros((len(metrics), 5))
        var_met  = np.zeros_like(mean_met)
        for r in rois:
            mean_met [i, :] = np.mean(metrics[r], axis=0)
            var_met  [i, :] = np.var (metrics[r], axis=0)
            i += 1

    print ('-----------------------------------------------')
    print 'Whole Majority voting'
    pred = np.zeros(len(y), dtype=int)
    for i in np.arange(len(y)):
        pred[i] = majority(preds[:, i])

    print_pred_metrics (y, pred)

    print ('-----------------------------------------------')
    print 'AUC 20 best ROIs'
    #calculate prediction based on nr ROIS with best training AUC
    nr = 20
    pred = np.zeros(len(y), dtype=int)

    for i in np.arange(len(y)):
        tauc    = taucs[:, np.arange(len(y)) != i]
        mtauc   = tauc.mean(axis=1)
        otauc   = mtauc.argsort()
        otauc   = otauc[otauc != 0]
        pred[i] = majority(preds[otauc[-nr:], i])

    print_pred_metrics (y, pred)

    print ('-----------------------------------------------')
    print 'F1-scores 20 best ROIs'
    #calculate prediction based on nr ROIS with best training f1-score
    nr = 20
    pred = np.zeros(len(y), dtype=int)

    for i in np.arange(len(y)):
        tf1     = tfones[:, np.arange(len(y)) != i]
        mtf1    = tf1.mean(axis=1)
        otf1    = mtf1.argsort()
        otf1    = otf1[otf1 != 0]
        pred[i] = majority(preds[otf1[-nr:], i])

    print_pred_metrics (y, pred)

    #calculate prediction based on CDR1 Alzheimer ROI List
    print ('-----------------------------------------------')
    print 'Selecting ROIs from CDR1 Alhzeimer ROI List'
    c1roisf = 'aal_allvalues_Alzheimer_CDR1.txt'
    c1roisp = os.path.join(wd, c1roisf)
    c1dx    = get_rois_indexes (c1roisp, rois)

    pred = np.zeros(len(y), dtype=int)
    for i in np.arange(len(y)):
        pred[i] = majority(preds[c1dx, i])

    print_pred_metrics (y, pred)

    #calculate prediction based on Mild Alzheimer ROI List
    print ('-----------------------------------------------')
    print 'Selecting ROIs from Mild Alzheimers ROI List'
    miroisf = 'aal_allvalues_Alzheimer_Mild.txt'
    miroisp = os.path.join(wd, miroisf)
    mildx   = get_rois_indexes (miroisp, rois)

    pred = np.zeros(len(y), dtype=int)
    for i in np.arange(len(y)):
        pred[i] = majority(preds[mildx, i])

    print_pred_metrics (y, pred)


    #calculate prediction based on full Alzheimer ROI List
    print ('-----------------------------------------------')
    print 'Selecting ROIs from Alhzeimer ROI List'
    alroisf = 'aal_allvalues_Alzheimer.txt'
    alroisp = os.path.join(wd, alroisf)
    alldx   = get_rois_indexes (alroisp, rois)

    pred = np.zeros(len(y), dtype=int)
    for i in np.arange(len(y)):
        pred[i] = majority(preds[alldx, i])

    print_pred_metrics (y, pred)

    print ('***********************************************')

#listing ROIs for each case
c1roilst = np.loadtxt(c1roisp, dtype=str)
miroilst = np.loadtxt(miroisp, dtype=str)
alroilst = np.loadtxt(alroisp, dtype=str)
for r in c1roilst[:,1]: print r
for r in miroilst[:,1]: print r
for r in alroilst[:,1]: print r

#--------------------------------------------------------------------
#showing best AUC regions
def save_nibabel (ofname, vol, affine, header=None):
   #saves nifti file
   ni = nib.Nifti1Image(vol, affine, header)
   nib.save(ni, ofname)


wd    = '/home/alexandre/Dropbox/Documents/phd/work/oasis_aal'
aalf  = os.path.join(wd, 'aal_MNI_1mm_FSL.nii.gz')
aalv  = nib.load(aalf).get_data()
aalh  = nib.load(aalf).get_header()
aalaf = nib.load(aalf).get_affine()

#best 20 F1-score
nr = 20
aalvals = np.unique(aalv)
fone_count = np.zeros(len(aalvals), dtype=int)

for i in np.arange(len(y)):
    tf1     = tfones[:, np.arange(len(y)) != i]
    mtf1    = tf1.mean(axis=1)
    otf1    = mtf1.argsort()
    otf1    = otf1[otf1 != 0]
    fone_count [otf1[-nr:]] += 1

aalfonedx   = aalvals[fone_count > 0]
aalfoneroiv = np.zeros_like(aalv)
for r in aalfonedx:
    aalfoneroiv[aalv == r] = r

save_nibabel (os.path.join(wd, 'aal_MNI_1mm_F1_20best.nii.gz'), aalfoneroiv, aalaf, aalh)

#best 20 AUC
nr = 20
aalvals = np.unique(aalv)
auc_count = np.zeros(len(aalvals), dtype=int)

for i in np.arange(len(y)):
    tauc    = taucs[:, np.arange(len(y)) != i]
    mtauc   = tauc.mean(axis=1)
    otauc   = mtauc.argsort()
    otauc   = otauc[otauc != 0]
    auc_count [otauc[-nr:]] += 1

aalaucdx   = aalvals[auc_count > 0]
aalaucroiv = np.zeros_like(aalv)
for r in aalaucdx:
    aalaucroiv[aalv == r] = r

save_nibabel (os.path.join(wd, 'aal_MNI_1mm_AUC_20best.nii.gz'), aalaucroiv, aalaf, aalh)


#listing Best AUC ROIs
for r in roilst[aalaucdx,1]: print r

#-------------------------------------------------------------------------------
def main(argv=None):



#-------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())


