#!/usr/bin/python

import os
import re
import sys
import argparse
import numpy as np
import nibabel as nib
import pickle

import scipy.stats as stats

#data preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#feature selection
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import zero_one

#other decompositions
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.lda import LDA

#pipeline
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

#scores
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report

#other decompositions
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.lda import LDA
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

#pipelining
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion

#debugging
from IPython.core.debugger import Tracer; debug_here = Tracer()

#visualization
import matplotlib.pyplot as plt
from itertools import product
from sklearn.decomposition import RandomizedPCA
from sklearn.lda import LDA
from sklearn.utils import shuffle

sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/aizkolari')
import aizkolari_utils as au

#bash
'''
d='/home/alexandre/Dropbox/Documents/phd/work/oasis_aal'

#wd='/media/alexandre/data/oasis_aal'

ft="jacs smoothmodgm norms modulatedgm geodan trace"
es="svm cart rf"
es="linsvm sgd percep"
es="svm"

#wd='/media/alexandre/toshiba/oasis_aal'
#fs="univariate fdr fpr extratrees pca rpca lda rfe rfecv"; fsf="none"
#n_cpus=2

wd='/scratch/oasis_aal'
fs="stats"; fsf="stats"
n_cpus=3

#cvfold=10
cvfold=loo

for e in $es; do
    for s in $fs; do
        for t in $ft; do
            subjlstf=${wd}/${t}_lst
            datadir=${wd}/oasis_${t}_${fsf}
            of="test_${cvfold}_${e}_${t}_${s}.pickle"
            if [ ! -f "${d}/${of}" ]; then
                echo $e - $t - $s
                ${d}/do_aal_stats_classification.py -s $subjlstf -o ${d} -d $datadir --fsmethod $s -f $t -e $e -c ${n_cpus} --cvfold ${cvfold}
            else
                echo ${of} already done!
            fi;
        done;
    done;
done;
'''

#feats     = "jacs"
#estimator = "tree"
#ncpus     = 3


#-------------------------------------------------------------------------------
def set_parser():

    ftypes     = ['jacs','smoothmodgm','geodan','modulatedgm','norms','trace']
    clfmethods = ['cart', 'gmm', 'rf', 'svm']
    fsmethods  = ['stats', 'rfe', 'rfecv', 'univariate', 'fdr', 'fpr', 'extratrees', 'pca', 'rpca', 'lda'] #svmweights

    parser = argparse.ArgumentParser(description='OASIS AAL classification experiment.')
    parser.add_argument('-s', '--subjlstf',  dest='subjlstf', default='', required=True,   help='list file with the subjects for the analysis. Each line: <class_label>,<subject_file>')
    parser.add_argument('-d', '--datadir',   dest='datadir', default='', required=True, help='data directory path')
    parser.add_argument('-o', '--outdir',    dest='outdir',   default='', required=False,  help='output data directory path. Will use datadir if not set.')
    parser.add_argument('-f', '--feats',     dest='feats', default='jacs', choices=ftypes, required=True, help='deformation measure type')
    parser.add_argument('--fsmethod',        dest='fsmethod', default='stats', choices=fsmethods, required=True, help='feature extraction method used to build the datasets')
    parser.add_argument('--cvfold',    dest='cvfold', default='10', choices=['10', 'loo'], required=False, help='Cross-validation folding method: stratified 10-fold or leave-one-out.')
    parser.add_argument('-e', '--estim',     dest='estimator', default='svm', choices=clfmethods, required=False, help='classifier type')
    parser.add_argument('-c', '--ncpus',     dest='ncpus', required=False, type=int, default=1, help='number of cpus used for parallelized grid search')
    parser.add_argument('-v', '--verbosity', dest='verbosity', required=False, type=int, default=2, help='Verbosity level: Integer where 0 for Errors, 1 for Input/Output, 2 for Progression reports')

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
def get_fsmethod (fsmethod, n_feats, n_subjs, n_jobs=1):

    if fsmethod == 'stats':
        return 'stats', None

    #Feature selection procedures
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
    fsmethods = { 'rfe'       : RFE(estimator=SVC(kernel="linear"), step=0.05, n_features_to_select=2),
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
                  'rfecv'     : RFECV(estimator=SVC(kernel="linear"), step=0.05, loss_func=zero_one), #cv=3, default; cv=StratifiedKFold(n_subjs, 3)
                                #Univariate Feature selection: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html
                  'univariate': SelectPercentile(f_classif, percentile=5),
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html
                  'fpr'       : SelectFpr (f_classif, alpha=0.05),
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html
                  'fdr'       : SelectFdr (f_classif, alpha=0.05),
                                #http://scikit-learn.org/stable/modules/feature_selection.html
                  'extratrees': ExtraTreesClassifier(n_estimators=50, max_features='auto', compute_importances=True, n_jobs=n_jobs, random_state=0),

                  'pca'       : PCA(n_components='mle'),
                  'rpca'      : RandomizedPCA(random_state=0),
                  'lda'       : LDA(),
    }

    #feature selection parameter values for grid search
    max_feats = ['auto']
    if n_feats < 10:
        feats_to_sel = range(2, n_feats, 2)
        n_comps = range(1, n_feats, 2)
    else:
        feats_to_sel = range(2, 20, 4)
        n_comps = range(1, 30, 4)
    max_feats.extend(feats_to_sel)

    fsgrid =    { 'rfe'       : dict(estimator_params = [dict(C=0.1), dict(C=1), dict(C=10)], n_features_to_select = feats_to_sel),
                  'rfecv'     : dict(estimator_params = [dict(C=0.1), dict(C=1), dict(C=10)]),
                  'univariate': dict(percentile = [1, 3, 5, 10]),
                  'fpr'       : dict(alpha = [1, 3, 5, 10]),
                  'fdr'       : dict(alpha = [1, 3, 5, 10]),
                  'extratrees': dict(n_estimators = [1, 3, 5, 10, 30, 50], max_features = max_feats),
                  'pca'       : dict(n_components = n_comps.extend(['mle']), whiten = [True, False]),
                  'rpca'      : dict(n_components = n_comps, iterated_power = [3, 4, 5], whiten = [True, False]),
                  'lda'       : dict(n_components = n_comps)
    }

    return fsmethods[fsmethod], fsgrid[fsmethod]

#-------------------------------------------------------------------------------
def parse_subjects_list (fname, datadir=''):
    labels = []
    subjs  = []

    if datadir:
        datadir += os.path.sep

    try:
        f = open(fname, 'r')
        for s in f:
            line = s.strip().split(',')
            labels.append(np.float(line[0]))
            subjf = line[1].strip()
            if not os.path.isabs(subjf):
                subjs.append (datadir + subjf)
            else:
                subjs.append (subjf)
        f.close()

    except:
        au.log.error( "Unexpected error: ", sys.exc_info()[0] )
        debug_here()
        sys.exit(-1)

    return [labels, subjs]

#-------------------------------------------------------------------------------
def shelve_vars (ofname, varlist):
   mashelf = shelve.open(ofname, 'n')

   for key in varlist:
      try:
         mashelf[key] = globals()[key]
      except:
         au.log.error('ERROR shelving: {0}'.format(key))

   mashelf.close()

#-------------------------------------------------------------------------------
def append_to_keys (mydict, preffix):
    return {preffix + str(key) : (transform(value) if isinstance(value, dict) else value) for key, value in mydict.items()}

#-------------------------------------------------------------------------------

def visualize_data (datafile):
    #pca = RandomizedPCA(n_components=2)
    au.log.info ('Linear Discriminant analysis')
    lda = LDA(n_components=2)
    fig, plots = plt.subplots(4, 4)
    fig.set_size_inches(50, 50)
    plt.prism()
    for i, j in product(xrange(4), repeat=2):
        if i > j:
            continue
        if i == j:
            continue

        X_ = X[(y == i) + (y == j)]
        y_ = y[(y == i) + (y == j)]

        #marks
        #marks = y_.astype(str)
        #marks[y_ == 0] = 'x'
        #marks[y_ == 1] = 'o'
        #marks[y_ == 2] = 'D'
        #marks[y_ == 3] = '1'

        #colors
        colors = y_.copy()
        colors[y_ == 0] = 0
        colors[y_ == 1] = 1
        colors[y_ == 2] = 2
        colors[y_ == 3] = 3

        #transform
        #X_trans = pca.fit_transform(X_)
        X_trans = lda.fit(X_, y_).transform(X_)

        #plots
        plots[i, j].scatter(X_trans[:, 0], X_trans[:, 1], c=colors, marker='o')
        plots[i, j].set_xticks(())
        plots[i, j].set_yticks(())

        plots[j, i].scatter(X_trans[:, 0], X_trans[:, 1], c=colors, marker='o')
        plots[j, i].set_xticks(())
        plots[j, i].set_yticks(())
        if i == 0:
            plots[i, j].set_title (j)
            plots[j, i].set_ylabel(j)

        #plt.scatter(X_trans[:, 0], X_trans[:, 1], c=y_)

    plt.tight_layout()
    plt.savefig(outfile)
#-------------------------------------------------------------------------------

def main(argv=None):

    parser  = set_parser()

    try:
       args = parser.parse_args ()
    except argparse.ArgumentError, exc:
       print (exc.message + '\n' + exc.argument)
       parser.error(str(msg))
       return -1

    subjlstf  = args.subjlstf.strip()
    datadir   = args.datadir.strip()
    feats     = args.feats.strip()
    outdir    = args.outdir.strip()
    clfmethod = args.estimator.strip()
    fsname    = args.fsmethod.strip()
    cvfold    = args.cvfold.strip()
    n_cpus    = args.ncpus
    verbose   = args.verbosity
    
    au.setup_logger(verbose, logfname=None)

    scale = True

    #label values
    n_class = 2

    #labels
    y, subjs = parse_subjects_list (subjlstf)
    scores = np.array(y)
    y      = np.array(y)
    y[scores > 0] = 1
    y = y.astype(int)

    n_subjs = len(subjs)

    #feature sets files
    fsregex = 'none'
    if fsname == 'stats':
        fsregex = 'stats'

    fsfiles = dir_match("(.)+_" + fsregex + "_(.)+.npy", datadir)
    fsfiles.sort()

    #results
    results = {}
    results['subjs']   = subjs
    results['y']       = y
    results['fsfiles'] = fsfiles

    #do it
    #fsfiles = [fsfiles[0]]
    for f in fsfiles:
        visualize_data (f)

        #roinom = str.split(str.split(f, ".")[-1], ".")[0]
        #roinom = str.split(str.split(f, ".")[0],"_")[-1]
        roinom = str.split(f.replace("oasis_" + feats + "_" + fsregex + "_", ""), ".")[0]

        print (roinom)

        #classification method instance
        data = np.load(os.path.join(datadir, f))
        n_subjs = data.shape[0]
        n_feats = data.shape[1]

        classif, clp = get_clfmethod (clfmethod, n_feats, n_subjs, n_cpus)

        #feature selection method instance
        fsmethod, fsp = get_fsmethod (fsname, n_feats, n_subjs, n_cpus)

        #results variables
        preds   = {}
        truth   = {}
        rscore  = {} #np.zeros(n_subjs) #ROI weights, based on AUC
        f1score = {} #np.zeros(n_subjs) #ROI weights, based on F1-score
        probs   = {} #np.zeros((n_subjs, n_class))
        best_p  = {}

        #cross validation
        if cvfold == '10':
            cv = StratifiedKFold(y, 10)
        elif cvfold == 'loo':
            cv = LeaveOneOut(len(y))

        fc = 0
        for train, test in cv:
            print '.',

            #train and test sets
            try:
                X_train, X_test, y_train, y_test = data[train,:], data[test,:], y[train], y[test]
                sc_train, sc_test = scores[train], scores[test]
            except:
                debug_here()

            #scaling
            if clfmethod == 'svm' or clfmethod == 'linsvm' or clfmethod == 'sgd':
                #scale_min = -1
                #scale_max = 1
                #[X_train, dmin, dmax] = au.rescale (X_train, scale_min, scale_max)
                #[X_test,  emin, emax] = au.rescale (X_test,  scale_min, scale_max, dmin, dmax)
                scaler  = StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test  = scaler.transform(X_test)


            #classifier instance
            elif clfmethod == 'gmm':
                classif.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                 for i in xrange(n_class)])

            #creating grid search pipeline
            if fsname != 'stats':
                #fsp   = append_to_keys(fsp, fsname + '__')
                pipe   = Pipeline([ ('fs', fsmethod), ('cl', classif) ])
                clap   = append_to_keys(clp, 'cl__')
                fisp   = append_to_keys(fsp, 'fs__')
                params = dict(clap.items() + fisp.items())
                gs     = GridSearchCV (pipe, params, n_jobs=n_cpus, verbose=0)
            else:
                gs     = GridSearchCV (classif, clp, n_jobs=n_cpus, verbose=0)

            if fsname == 'univariate':
                gs.fit(X_train, sc_train)
            else:
                gs.fit(X_train, y_train)

            #predictions and best parameters
            preds [fc] = gs.predict(X_test)

            #AUC score based on training classification
            roc_auc = 0
            if gs.predict_proba:
                rprobs = gs.predict_proba(X_train)

                rfpr, rtpr, rthresholds = roc_curve(y_train, rprobs[:, 1], 1)
                roc_auc = auc(rfpr, rtpr)

            rscore[fc] = roc_auc

            #save results
            best_p[fc] = gs.best_params_
            preds [fc] = gs.predict(X_test)
            truth [fc] = y_test

            if gs.predict_proba:
                probs [fc] = gs.predict_proba(X_test)

            #classification_report(y, preds)

            fc += 1

        #results[roinom] = classification_metrics (y, preds, probs)
        results[roinom] = {}
        results[roinom]['clfmethod']        = clfmethod
        results[roinom]['cv']               = cv
        results[roinom]['cvgrid']           = clp
        results[roinom]['preds']            = preds
        results[roinom]['truth']            = truth
        results[roinom]['probs']            = probs
        results[roinom]['best_params']      = best_p
        results[roinom]['train_auc_scores'] = rscore
        results[roinom]['train_f1_scores']  = f1score

    #saving results
    if not outdir:
        outdir = datadir

    outfname = os.path.join(outdir, 'test_' + cvfold + '_' + clfmethod + '_' + feats + '_' + fsname)

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


