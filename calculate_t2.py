from __future__ import print_function, division

import sys
sys.path.insert(0, 'lib')
import numpy as np
import random
import scipy.io as sio
import os
import pandas as pd
import scipy.ndimage as ndimage
import math
import os
import scipy.linalg as la
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from skimage import measure
import scipy.stats as ss
import skimage


#########################################################
# Calculating T2 Values for Segmented Voxels
#########################################################

def exp_func(mri_time, A, m, b):
    return A*np.exp(-m*mri_time)

def fit_t2(t2imgs, t2times, segmentation = None, n_jobs = 4, show_bad_pixels = True):
    
    '''
    Fits T2 curves to the T2_weighted images in each slice.
    IN:
        t2imgs - with T2 weighted images in numpy array (nr_slices, time_steps, width, heigth)
        t2times - list with aquisition times
        segmentation - segmentation matrix (nr_slices, width, heigth)
        n_jobs - number of parallel jobs
    OUT:
        matrix (nr_slices, width, heigth) with T2 values
    '''
    t2_tensor = np.zeros((t2imgs.shape[0], t2imgs.shape[2], t2imgs.shape[3]))

    def fit_per_slice(slice_idx, show_bad_pixels):
        scan = t2imgs[slice_idx,:,:,:]
        
        mri_time = np.array(t2times[slice_idx]) - t2times[slice_idx][0] #np.array(t2times[slice_idx])#
        
        if not segmentation is None: # if we have a segmentation
            segmentation_mask = segmentation[slice_idx,:,:]
            (cartilage_indices_r, cartilage_indices_c) = np.where(segmentation_mask)

        t2_matrix = np.full((scan.shape[1], scan.shape[2]), np.nan)        
        if len(cartilage_indices_r)> 0:
            for i in np.arange(len(cartilage_indices_r)):
                ir = cartilage_indices_r[i]
                ic = cartilage_indices_c[i]
                               
                if all(scan[:,ir,ic] == scan[0,ir,ic]): # if constant value, decay is 0 
                    continue
                    
                try:
                    parameters,_ = curve_fit(exp_func, 
                                    mri_time[1:], 
                                    scan[1:,ir,ic], 
                                    p0 = [scan[0,ir,ic], .03, 0])#, 
#                                     bounds = ([-np.inf, 0, -np.inf], [np.inf, 100, np.inf]))
                    m = parameters[1]
                    t2_ = 1./m
                    t2_matrix[ir, ic] = t2_
                    if show_bad_pixels:
                        if ((t2_ > .100) or (t2_< -.100)): 
                            print(t2_)
                            plt.plot(mri_time, scan[:,ir,ic])
                            plt.plot(mri_time, exp_func(mri_time, *parameters), 'r-')
                            plt.show()
                        
            
                except RuntimeError:
                    if show_bad_pixels:
                        plt.plot(mri_time, scan[:,ir,ic])
                        plt.title("Did not converge")
                        plt.show()                   

        return t2_matrix

    for i in range(t2imgs.shape[0]):
        t2_tensor[i,:,:] = fit_per_slice(i, show_bad_pixels)*1000 # in ms
    return t2_tensor