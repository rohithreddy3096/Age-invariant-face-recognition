import numpy as np
from numpy.core.umath_tests import inner1d



# Hausdorff Distance
def HausdorffDist(A,B):

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    # Find DH
    dH = np.max(np.array([np.max(np.min(D_mat,axis=0)),np.max(np.min(D_mat,axis=1))]))
    return(dH)

def ModHausdorffDist(A,B):

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    
    FHD = np.mean(np.min(D_mat,axis=1))
    
    RHD = np.mean(np.min(D_mat,axis=0))
    
    MHD = np.mean(np.array([FHD, RHD]))
    return MHD
