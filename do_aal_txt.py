#!/usr/bin/python

import os
import numpy as np

rootdir  = '/media/data/oasis_aal'

#create a txt file combining both aal files
aal1f = rootdir + os.path.sep + 'aal.txt'
aal2f = rootdir + os.path.sep + 'aal.nii.txt'
aal3f = rootdir + os.path.sep + 'aal_allvalues.txt'

aal1 = np.loadtxt(aal1f, dtype=str)
aal2 = np.loadtxt(aal2f, dtype=str)

m = aal1.shape[0]
n = aal1.shape[1]

aal  = np.zeros((m,n+1), dtype=aal1.dtype)

aal[:,0:3] = aal1
aal[:,3] = aal2[:,0]

np.savetxt (aal3f, aal, fmt='%s')

