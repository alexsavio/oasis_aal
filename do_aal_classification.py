#!/usr/bin/python

import os
import re
import sys
import argparse
import numpy as np
import nibabel as nib
import pickle

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
#        echo $f - $e
#        ${d}/do_aal_classification.py -f $f -e $e -c 3
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

def main(argv=None):

    parser  = set_parser()

    try:
       args = parser.parse_args ()
    except argparse.ArgumentError, exc:
       print (exc.message + '\n' + exc.argument)
       parser.error(str(msg))
       return -1

    feats     = args.feats.strip()
    estimator = args.estimator.strip()
    ncpus     = args.ncpus

    #feats = 'jacs'
    #feats = 'smoothmodgm'
    #feats = 'geodan'
    #feats = 'norms'
    #feats = 'trace'

    #ftype = 'raw'
    ftype  = 'stats'
    nfeats = 7

    #number of threads to use in grid search
    #ncpus = 4

    #estimator type
    #estimator = 'tree'
    #estimator = 'rf'
    #estimator = 'svm'

    #SVM kernel
    #kernel = 'linear'
    #kernel = 'rbf'
    #both included in gridsearch

    #label values
    nclass = 2
    control_label = 0
    patient_label = 1

    #work folders
    rootdir  = '/media/data/oasis_aal'
    #rootdir  = '/home/alexandre/Dropbox/Documents/phd/work/oasis_aal'
    workdir  = rootdir + os.path.sep + 'oasis_' + feats + '_feats'

    #subject list
    subjlstf = workdir + os.path.sep + 'oasis_' + feats + '_' + ftype + '_subjlist.txt'
    subjlst  = np.loadtxt(subjlstf, dtype=str)
    nsubjs   = len(subjlst)
    labels   = np.zeros(nsubjs, dtype=int)
    for i in np.arange(nsubjs):
        s = subjlst[i]
        if s.find('control'):
            labels[i] = control_label
        else:
        #elif s.find('patient'):
            labels[i] = patient_label

    #feature sets files
    searchre = "(.)+.npy"
    if ftype == 'stats':
        searchre = "(.)+.npy"

    fsfiles = dir_match("(.)+_stats_(.)+.npy", workdir)
    fsfiles.sort()

    # Set the parameters for Grid search
    # SVM
    cvals = np.logspace(-3, 3, num=7, base=10)
    gvals = np.logspace(-3, 3, num=7, base=10)
    svm_parameters = [{'kernel': ['rbf'], 'C': cvals, 'gamma': gvals}] #,
    #                    {'kernel': ['linear'], 'C': cvals}]

    #RF
    ntrees = [10, 25, 50, 100, 200]
    rf_parameters = [{'n_estimators': ntrees}]

    #available scores
    scores = {'zero_one': zero_one_score, 
              'precision' : precision_score,
              'recall': recall_score}

    #results
    results = {}
    results['labels'] = labels

    #do it
    #fsfiles = [fsfiles[0]]
    for f in fsfiles:
        #roinom = str.split(str.split(f, ".")[-1], ".")[0]
        #roinom = str.split(str.split(f, ".")[0],"_")[-1]
        roinom = str.split(f.replace("oasis_" + feats + "_" + ftype + "_", ""), ".")[0]

        print (roinom)

        data = np.load(workdir + os.path.sep + f)
        idxs = np.arange(nsubjs)

        #OPTION 1: scikit-learn
        preds   = np.zeros(nsubjs)
        rscore  = np.zeros(nsubjs) #ROI weights, based on AUC
        f1score = np.zeros(nsubjs) #ROI weights, based on F1-score
        probs   = np.zeros((nsubjs, nclass))

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []

        loo = LeaveOneOut(nsubjs)
        for train, test in loo:
            #print (test)

            #training data
            rd = data  [train]
            rl = labels[train]

            #test data
            sd = data  [test]
            sl = labels[test]

            #scaling
            scaler = preprocessing.Scaler().fit(rd)
            rd_scaled = scaler.transform(rd)
            sd_scaled = scaler.transform(sd)

            #Creating the base classifier
            #clf = svm.SVC(kernel='linear', probability=True)
            #clf = svm.SVC(kernel='linear', probability=True, class_weight='auto')
            #clf = svm.SVC(kernel=kernel, probability=True, class_weight='auto')
            #clf = svm.SVC()
            #clf = clf.fit(rd_scaled, rl)

            #recursive feature elimination RFE
            #rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(labels, 2),
            #loss_func=zero_one)
            #rfecv.fit(rd_scaled, rl)

            if estimator == 'svm':
                #GRID SEARCH
                #SVM
                clf = GridSearchCV(svm.SVC(C=1, probability=True), svm_parameters, score_func=zero_one_score, n_jobs=ncpus)

            elif estimator == 'rf':
                clf = GridSearchCV(RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=None), rf_parameters, score_func=zero_one_score, n_jobs=ncpus)

            if estimator == 'svm' or estimator == 'rf':
                clf.fit(rd_scaled, rl, cv=4)

                #TRAIN AND TEST
                clf = clf.best_estimator_

                clf.fit(rd_scaled, rl)

            if estimator == 'tree':
                #Classification Trees
                clf = tree.DecisionTreeClassifier(criterion='gini')
                clf.fit(rd_scaled, rl)

            #print(clf)

            #mean F1-score base on 10-fold CV on training set
            f1scores = cross_validation.cross_val_score(clf, rd_scaled, rl, cv=10, score_func=metrics.f1_score)
            f1score[test] = np.mean(f1scores)

            #AUC score based on training classification
            rprobs = clf.predict_proba(rd_scaled)
            rfpr, rtpr, rthresholds = roc_curve(rl, rprobs[:, 1])
            roc_auc = auc(rfpr, rtpr)
            rscore[test] = roc_auc

            #save results
            preds [test] = clf.predict(sd_scaled)
            probs [test] = clf.predict_proba(sd_scaled)

            classification_report(labels, preds)

        results[roinom] = classification_metrics (labels, preds, probs)
        results[roinom]['clf']              = clf
        results[roinom]['preds']            = preds
        results[roinom]['probs']            = probs
        results[roinom]['train_auc_scores'] = rscore
        results[roinom]['train_f1_scores']  = f1score

    outfname = rootdir + os.path.sep + 'test_' + estimator + '_' + feats
    #np.savez (outfname + '.npz', results)
    #np.save  (outfname + '.npy', results)
    of = open(outfname + '.pickle', 'w')
    pickle.dump (results, of)
    of.close()

    #inf = open(outfname + '.pickle', 'r')
    #res = pickle.load(inf)

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())


