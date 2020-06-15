import numpy as np

#########################################################
# Refining Initial Segmentations and T2 maps
#########################################################
def t2_threshold(seg_initial, t2_initial, t2_low=0, t2_high=100):
    '''
    Removes segmented voxels that are ourside the physiologic range of cartilage from the segmentation mask and T2 map
    If seg_initial is probabilistic, the returned seg will be too.
    
    seg_initial (3D numpy array of shape (#slices,H,W)): volumetric segmentation mask. 
    t2_initial (3D numpy array of shape (#slices,H,W): volumetric t2 map. Only initially segmented cartilage voxels are non-NaN
    
    '''
    t2 = np.copy(t2_initial)
    t2[np.isnan(t2)]=0
    t2[t2 > t2_high] = 0
    t2[t2 < t2_low] = 0
    
    seg = seg_initial * (1*(t2>0))
    
    return seg, t2

def optimal_binarize(seg_probabilistic, t2_probabilistic, prob_threshold, voxel_count_threshold):
    '''
    seg_probabilistic (3D numpy array): probabilistic segmentation mask produced by the model and refined by t2_threshold function
    
    t2_probabilistic (3D numpy array): t2 map with non-zero values at each location where seg_probabilistic has a non-zero value
    
    prob_threshold (float): minimum probability output from the model that should be included in the cartilage segmentation
    
    voxel_count threshold (int): minimum number of segmented cartilage voxels that a slice needs in order for the slice's voxels to be retained in the segmentation
    
    '''
    
    seg_binary = (seg_probabilistic > prob_threshold)*1
    seg_binary[np.sum(seg_binary, axis = (1,2)) < voxel_count_threshold] = 0
    seg_binary = seg_binary.squeeze()
    t2_binary = t2_probabilistic * seg_binary
    return seg_binary, t2_binary

