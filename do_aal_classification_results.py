#!/usr/bin/python

import os
import re
import sys
import argparse
import numpy as np
import nibabel as nib

import scipy.stats as stats

from sklearn import svm
from sklearn import tree
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import zero_one_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

from IPython.core.debugger import Tracer; debug_here = Tracer()

sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/aizkolari')
import aizkolari_utils as au
import aizkolari_export as ae
import aizkolari_svmperf as asvm

#bash
#d='/home/alexandre/Dropbox/Documents/phd/work/oasis_aal'
#fs="jacs smoothmodgm geodan norms trace"
#es="tree rf svm"
#for e in $es; do
#    for f in $fs; do
#        if [ ! -f ${d}/"test_${e}_${f}.pickle" ]; then
#            echo $f - $e
#            ${d}/do_aal_classification.py -f $f -e $e -c 3
#        fi
#    done;
#done;

#feats     = "jacs"
#estimator = "tree"
#ncpus     = 3


#-------------------------------------------------------------------------------
def set_parser():
    parser = argparse.ArgumentParser(description='OASIS AAL classification experiment.')
    parser.add_argument('-f', '--feats', dest='feats', default='jacs', choices=['jacs','smoothmodgm','geodan','norms','trace'], required=True, help='deformation measure type')
    parser.add_argument('-e', '--estim', dest='estimator', default='svm', choices=['svm','tree','rf'], required=False, help='classifier type')
    parser.add_argument('-c', '--ncpus', dest='ncpus', required=False, type=int, default=1, help='number of cpus used for parallelized grid search')

    return parser

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
def classification_metrics (targets, preds, probs):
    fpr, tpr, thresholds = roc_curve(targets, probs[:, 1])
    roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(targets, preds)

    #accuracy
    accuracy = zero_one_score(targets, preds)

    #recall? True Positive Rate or Sensitivity or Recall
    recall = recall_score(targets, preds)

    #precision
    precision = precision_score(targets, preds)

    tnr = 0.0
    #True Negative Rate or Specificity?
    if len(cm) == 2:
        tnr = float(cm[0,0])/(cm[0,0] + cm[0,1])

    out = {}
    out['accuracy'] = accuracy
    out['recall'] = recall
    out['precision'] = precision
    out['tnr'] = tnr
    out['roc_auc'] = roc_auc

    return out

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

wd = '/media/data/oasis_aal'
wd = '/home/alexandre/Dropbox/Documents/phd/work/oasis_aal'

npylst = dir_search ('.npy', wd)

for f in npylst:
    res = np.load(wd + os.path.sep + f)

    rois = res.keys()
    for i in rois:
        if i == 'labels': continue
        
        

        results[roinom] = classification_metrics (labels, preds, probs)
        results[roinom]['clf']              = clf
        results[roinom]['preds']            = preds
        results[roinom]['probs']            = probs
        results[roinom]['train_auc_scores'] = rscore
        results[roinom]['train_f1_scores']  = f1score

    outfname = rootdir + os.path.sep + 'test_' + estimator + '_' + feats
    #np.savez (outfname + '.npz', results)
    np.save  (outfname + '.npy', results)

#-------------------------------------------------------------------------------
def main(argv=None):



#-------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())


