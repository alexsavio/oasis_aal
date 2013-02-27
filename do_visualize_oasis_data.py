#!/usr/bin/python

import os
import re
import sys
import argparse
import subprocess
import logging as log
import numpy as np
import nibabel as nib

sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/aizkolari')
import aizkolari_utils as au

sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/visualize_volume')
import visualize_volume as vis

sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/oasis_svm')
from do_classification_utils import *

#from IPython.core.debugger import Tracer; debug_here = Tracer()

import matplotlib.pyplot as plt
from itertools import product
from sklearn.decomposition import RandomizedPCA
from sklearn.lda import LDA
from sklearn.utils import shuffle

##===================================================================
##creating subject files for this experiment
'''
import os
import re
import numpy as np

def find (lst, regex):
  o = []
  for i in lst:
     if re.search (regex, i):
        o.append(i)
  return o

hn = au.get_hostname()
if hn == 'azteca':
   wd = '/data/oasis_jesper_features'
elif hn == 'corsair':
    wd = '/media/alexandre/toshiba/oasis_svm'
elif hn == 'hpmed':
    wd = '/home/alexandre/Desktop/oasis_svm'

#measures = ['jacs', 'norms', 'modulatedgm', 'trace', 'geodan']
measures = ['jacs', 'modulatedgm']

for m in measures:
    files = find(os.listdir(wd + os.path.sep + m), '.nii.gz')
    files = np.sort(files)
    lst = []
    for f in files:
        s = str(float(f[13:15])) + ',' + f
        lst.append(s)

    of = wd + os.path.sep + m + '_lst'
    print ('Saving ' + of)
    np.savetxt(of, np.array(lst), fmt='%s')
'''
##===================================================================

#-------------------------------------------------------------------------------

def set_parser():
    parser = argparse.ArgumentParser(description='Script for experiments')

    fsmethods  = ['rfe', 'rfecv', 'univariate', 'fdr', 'fpr', 'extratrees', 'none']

    parser.add_argument('-i', '--in', dest='infile', required=True, help='list file with the subjects for the analysis. Each line: <class_label>,<subject_file>')
    parser.add_argument('-o', '--out', dest='outfile', required=True, help='Python shelve output file name preffix.')
    parser.add_argument('-d', '--datadir', dest='datadir', required=False, help='folder path where the subjects are, if the absolute path is not included in the subjects list file.', default='')
    parser.add_argument('-m', '--mask', dest='mask', default='', required=False, help='Mask file to extract feature voxels, any voxel with values > 0 will be included in the extraction.')
    parser.add_argument('-f', '--fsmethod', dest='fsmethod', default='none',
choices=fsmethods, required=False, help='Feature selection method')

    parser.add_argument('-v', '--verbosity', dest='verbosity', required=False, type=int, default=2, help='Verbosity level: Integer where 0 for Errors, 1 for Input/Output, 2 for Progression reports')

#    parser.add_argument('--scale', dest='scale', default=False, action='store_true', required=False, 
#            help='This option will enable Range scaling of the training data.')
#    parser.add_argument('--scale_min', dest='scale_min', default=-1, type=int, required=False, help='Minimum value for the new scale range.')
#    parser.add_argument('--scale_max', dest='scale_max', default= 1, type=int, required=False, help='Maximum value for the new scale range.')

    return parser

#-------------------------------------------------------------------------------
def select_features (X, y, fsmethod = 'univariate', njobs = 3, nfeats=None):

    from sklearn.feature_selection import f_classif

    #Feature selection procedures
    ###############################################################################
    #TEST 3 - RFE and Classification
    #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
    if fsmethod == 'rfe':
        # Create the RFE object and rank each pixel
        from sklearn.svm import SVC
        from sklearn.feature_selection import RFE

        svc = SVC(kernel="linear", C=1)
        selector = RFE(estimator=svc, step=0.05, n_features_to_select=nfeats)

    ###############################################################################
    #TEST 3 - RFE with CV
    #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
    elif fsmethod == 'rfecv':
        # Create the RFE object and rank each pixel
        from sklearn.svm import SVC
        from sklearn.feature_selection import RFECV
        from sklearn.cross_validation import KFold
        from sklearn.metrics import zero_one

        svc = SVC(kernel="linear")
        selector = RFECV(estimator=svc, step=0.05, cv=KFold(len(y), 6),
                          loss_func=zero_one, n_features_to_select=nfeats)

    ###############################################################################
    #TEST 4 - Univariate Feature selection
    #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html
    # Univariate feature selection with F-test for feature scoring
    # We use the default selection function: the 10% most significant features
    elif fsmethod == 'univariate':
        from sklearn.feature_selection import SelectPercentile
        selector = SelectPercentile(f_classif, percentile=5)

    #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html
    elif fsmethod == 'fpr':
        from sklearn.feature_selection import SelectFpr
        selector = SelectFpr (f_classif, alpha=0.05)

    #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html
    elif fsmethod == 'fdr':
        from sklearn.feature_selection import SelectFdr
        selector = SelectFdr (f_classif, alpha=0.05)

    #svm weights
    #http://scikit-learn.org/stable/auto_examples/plot_feature_selection.html#example-plot-feature-selection-py
    #elif fsmethod == 'svmweights':
    #    from sklearn import svm
    #    selector = svm.SVC(kernel='linear')

    #trees feature selection
    #http://scikit-learn.org/stable/modules/feature_selection.html
    elif fsmethod == 'extratrees':
        n_jobs = 4
        from sklearn.ensemble import ExtraTreesClassifier
        selector = ExtraTreesClassifier(n_estimators=100,
                                       max_features=128,
                                       compute_importances=True,
                                       n_jobs=n_jobs,
                                       random_state=0)

    #fsmethods = np.array(['rfe', 'rfecv', 'univariate', 'fdr', 'fpr', 'svmweights', 'extratrees'])
    elif fsmethod == 'None':
        return None

    else:
        au.log.error ('ERROR: select_features: Not valid fsmethod: ' + fsmethod + '.')
        return

    selector.fit(X, y)

    return selector

#-------------------------------------------------------------------------------
#        if fsmethod == 'univariate' or fsmethod == 'fpr' or fsmethod == 'fdr':
#            sels = -np.log10(selector.scores_)
#        elif fsmethod == 'svmweights':
#            sels = (selector.coef_ ** 2).sum(axis=0)
#        elif fsmethod == 'rfe':
#            sels = selector.ranking_
#        elif fsmethod == 'extratrees':
#            sels = selector.feature_importances_
#        elif fsmethod == 'rfecv':
#            sels = selector.cv_scores_
#        sels = np.nan_to_num(sels)
#        sels /= sels.max()

#-------------------------------------------------------------------------------

def shelve_vars (ofname, varlist):
   mashelf = shelve.open(ofname, 'n')

   for key in varlist:
      try:
         mashelf[key] = globals()[key]
      except:
         log.error('ERROR shelving: {0}'.format(key))

   mashelf.close()

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
        sys.exit(-1)

    return [labels, subjs]

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def do_oasis_visualize_pca (args):

    subjsf     = args.infile.strip   ()
    outfile    = args.outfile.strip  ()
    datadir    = args.datadir.strip  ()
    maskf      = args.mask.strip     ()
    fsmethod   = args.fsmethod.strip ()

#    scale      = args.scale
#    scale_min  = args.scale_min
#    scale_max  = args.scale_max

    verbose    = args.verbosity

    #logging config
    au.setup_logger(verbose)

    #loading mask
    msk     = nib.load(maskf).get_data()
    nvox    = np.sum  (msk > 0)
    indices = np.where(msk > 0)

    #reading subjects list
    [scores, subjs] = parse_subjects_list (subjsf, datadir)
    scores = np.array(scores)

    imgsiz = nib.load(subjs[0]).shape
    nsubjs = len(subjs)

    #checking mask and first subject dimensions match
    if imgsiz != msk.shape:
        au.log.error ('Subject image and mask dimensions should coincide.')
        exit(1)

    #relabeling scores to integers, if needed
    if not np.all(scores.astype(np.int) == scores):
        unis = np.unique(scores)
        scs  = np.zeros (scores.shape, dtype=int)
        for k in np.arange(len(unis)):
            scs[scores == unis[k]] = k
        y = scs.copy()
    else:
        y = scores.copy()

    #loading data
    au.log.info ('Loading data...')
    X = np.zeros((nsubjs, nvox), dtype='float32')
    for f in np.arange(nsubjs):
        imf = subjs[f]
        au.log.info('Reading ' + imf)

        img = nib.load(imf).get_data()
        X[f,:] = img[msk > 0]

    #demo
    '''
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata("MNIST original")
    X, y = mnist.data[:60000] / 255., mnist.target[:60000]
    X, y = shuffle(X, y)
    X, y = X[:5000], y[:5000] # lets subsample a bit for a first impression
    '''

    #lets start plotting
    au.log.info ('Preparing plots...')
    X, y = shuffle(X, y)
    X = X/X.max()

    #reducing training and test data
    if fsmethod != 'none':
        au.log.info ('Feature selecion : ' + fsmethod)
        selector  = select_features (X, y, fsmethod)
        X = selector.transform(X)

    #au.log.info ('Randomized PCA')
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
## START MAIN
#-------------------------------------------------------------------------------
def main(argv=None):

    #parsing arguments
    parser = set_parser()

    try:
        args = parser.parse_args ()
    except argparse.ArgumentError, exc:
        print (exc.message + '\n' + exc.argument)
        parser.error(str(msg))
        return 0

    do_oasis_visualize_pca (args)


###############################################################################
#MAIN
if __name__ == "__main__":
    sys.exit(main())
