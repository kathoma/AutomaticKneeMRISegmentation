import scipy.ndimage as ndimage
import numpy as np
import matplotlib.pyplot as plt

def align_projections(proj1, proj2, pad_adjust = True, overlap_adjust = False):
    
    '''
    adds rows and columns where necessary in order to make two cartilage projections the same dimensions
    
    inputs:
    proj1 & proj2 (2D numpy arrays): cartilage projections
    
    pad_adjust (bool): If True, adds rows and columns where necessary so the two cartilage projections have the same dimensions
    
    overlap_adjust (bool): If True, it zeros any non-overlapping cartilage
    
    outputs:
    adjusted projections for both timepoints
    
    '''
    
    proj1[np.isnan(proj1)]=0
    proj2[np.isnan(proj2)]=0

    # Pad one image as necessary so that the two time points have same number of rows and columns
    if pad_adjust:
        
        # Throw out empty rows
        proj1 = proj1[np.sum(proj1, axis = (1))>0]
        proj2 = proj2[np.sum(proj2, axis = (1))>0]
        
        # Throw out empty columns
        proj1 = proj1[np.sum(proj1, axis = (1))>0]
        proj2 = proj2[np.sum(proj2, axis = (1))>0]
    
        while proj1.shape[0] != proj2.shape[0]:
            if proj1.shape[0] > proj2.shape[0]:
                top = np.sum(proj1[0]!=0)
                bottom = np.sum(proj1[-1]!=0)
                if top > bottom:
                    proj2 = np.concatenate([proj2, 
                                           np.zeros((1,proj2.shape[1]))]) # add to the bottom
                else:
                    proj2 = np.concatenate([np.zeros((1,proj2.shape[1])), 
                                           proj2]) # add to the top

            if proj2.shape[0] > proj1.shape[0]:
                top = np.sum(proj2[0]!=0)
                bottom = np.sum(proj2[-1]!=0)
                if top > bottom:
                    proj1 = np.concatenate([proj1, 
                                           np.zeros((1,proj1.shape[1]))]) # add to the bottom
                else:
                    proj1 = np.concatenate([np.zeros((1,proj1.shape[1])), 
                                           proj1]) # add to the top
                    
        
    
        while proj1.shape[1] != proj2.shape[1]:
            if proj1.shape[1] > proj2.shape[1]:
                left = np.sum(proj1[:,0]!=0)
                right = np.sum(proj1[:,-1]!=0)
                if left > right:
                    proj2 = np.concatenate([proj2, 
                                           np.zeros((proj2.shape[0],1))],
                                           axis = 1) # add to the right
                else:
                    proj2 = np.concatenate([np.zeros((proj2.shape[0],1)), 
                                           proj2],
                                           axis = 1) # add to the left

            if proj2.shape[1] > proj1.shape[1]:
                left = np.sum(proj2[:,0]!=0)
                right = np.sum(proj2[:,-1]!=0)
                if left > right:
                    proj1 = np.concatenate([proj1, 
                                           np.zeros((proj1.shape[0],1))],
                                           axis = 1) # add to the right
                else:
                    proj1 = np.concatenate([np.zeros((proj1.shape[0],1)), 
                                           proj1],
                                           axis = 1) # add to the left
                
    if overlap_adjust:            
        overlap = np.logical_and(proj1!=0, proj2!=0)
        proj1 = proj1*(overlap*1)
        proj2 = proj2*(overlap*1)
    
    return proj1, proj2

def match_shapes(mat_a, mat_b):
    """
    For two matrices where *mat_a* has more rows (first dim) it pads the
    *mat_b* with vectors of zero above and below.
    """
    assert mat_a.shape[0] >= mat_b.shape[0], "first matrix should be bigger"
    cols = mat_a.shape[1]
    zerosvec = np.zeros((1,cols))
    while mat_a.shape[0] - mat_b.shape[0] > 1:
        mat_b = np.r_[zerosvec, mat_b, zerosvec]
    if mat_a.shape[0] == mat_b.shape[0]:
        return mat_b
    assert mat_a.shape[0] - mat_b.shape[0] == 1, \
        'sth wrong in difference a:{} b:{}'.format(mat_a.shape[0], mat_b.shape[0])
    mat_b_up = np.r_[zerosvec, mat_b]
    mat_b_down = np.r_[zerosvec, mat_b]
    if np.sum((mat_a>0)*(mat_b_up>0)) > np.sum((mat_a>0)*(mat_b_down>0)):
        return mat_b_up
    else:
        return mat_b_down

def remove_empty_rows(image):
    "Removes empty rows from image (2d numpy.array)"
    mask = image == 0
    rows = np.flatnonzero((~mask).sum(axis=1))
    cropped = image[rows.min():rows.max()+1, :]
    return cropped

def eroded_and_mask(mata, matb, kernel = np.ones((10,10))):
    """
    Creatives mask for matrices *mata* and *matb* ( each 2D numpy.array)
    with positive elements.
    Erosion is applied first to each of matrix with *kernel* default(2x2)
    Then pointwise logical and is taken between these two masks.
    """
    assert mata.shape == matb.shape, "Shapes don't match"
    assert (mata>=0).all() and (matb>=0).all(), "Works only for positive matrices"
#     ermask_a = ndimage.morphology.binary_erosion(1*(mata > 0),
#                                     structure = kernel).astype(np.int)
#     ermask_b = ndimage.morphology.binary_erosion(1*(matb > 0),
#                                     structure = kernel).astype(np.int)
    
    overlap = (1*(mata > 0)) * (1*(matb > 0))
    eroded_mask = ndimage.morphology.binary_erosion(overlap, structure = kernel).astype(np.int)
    return eroded_mask
#     return ermask_a*ermask_b

def make_difference(t1_data, t2_data, save = None, plot = False):
    '''
    This fitting femoral cartilage to a cylinder, then binning every *angular_bin*
    degrees all pixels.
    IN:
        t1_data - matrix with encoded averaged values of cartilage T2 map (from projection)
                  time point 1
        t2_data - matrix with encoded averaged values of cartilage T2 map (from projection)
                  time point 2
        save    - if string, then it saves matrix to file from the path *save*, otherwise
                  it returns diffmat (default: None). Saved NPZ file will have two fields:
                  *diff* with difference between *t1_data* and *t2_data* and *mask* with 
                  a binary mask.
        plot    - flag for plotting - default False.
                  (if save is str is saves plot as PNG to *save* path, otherwise it shows)
    OUT:
        diffmat  - matrix with difference between t1_data and t2_data after matching
                   shape of matrices
    '''
    t1_proj = remove_empty_rows(t1_data)
    t2_proj = remove_empty_rows(t2_data)
    # in super and deep there's some empty slices, this handles it
    # but TODO: maybe account for it in projection?
    for i in np.argwhere(np.isnan(t1_proj)): 
        t1_proj[tuple(i)]=0
    for i in np.argwhere(np.isnan(t2_proj)):
        t2_proj[tuple(i)]=0
        
#     t1_proj, t2_proj = align_projections(t1_proj, t2_proj, pad_adjust = True, overlap_adjust = True)
    if t1_proj.shape != t2_proj.shape:
        if t1_proj.shape[0] < t2_proj.shape[0]:
            t1_proj = match_shapes(t2_proj, t1_proj)
        else:
            t2_proj = match_shapes(t1_proj, t2_proj)
    
    
    kernel_size = 15
    middle_r = (kernel_size/2)-.5
    middle_c = (kernel_size/2)-.5
    kernel = np.ones((kernel_size,kernel_size))
    for i in (range(kernel_size)):
        for j in (range(kernel_size)):
            if np.linalg.norm([(i-middle_r),(j-middle_c)]) > 4:
                kernel[i,j]=0
    binary_erosion_mask = eroded_and_mask(t1_proj, t2_proj, kernel)
    proj_diff = t2_proj - t1_proj
    diff_masked = proj_diff * binary_erosion_mask
    if save:
        np.savez(save, diff = diff_masked, mask = binary_erosion_mask)
    if plot:
        plt.figure(figsize=(13,2.5))
        plt.subplot(141)
        plt.imshow(-1*(t1_proj>0),vmin = -1,vmax = 1, cmap = 'seismic')
        plt.imshow(1*(t2_proj>0),alpha = 0.5, vmin = -1, vmax=1, cmap = 'seismic')
        plt.subplot(142)
        plt.imshow(binary_erosion_mask, vmin = -1,vmax = 1, cmap = 'seismic')
        plt.subplot(143)
        rg_ = np.max([abs(np.max(proj_diff)), abs(np.min(proj_diff))])
        plt.imshow(proj_diff, cmap = 'RdBu', vmin = -rg_, vmax = rg_)
        plt.colorbar()
        plt.subplot(144)
        rg_ = np.max([abs(np.max(diff_masked)), abs(np.min(diff_masked))])
        plt.imshow(diff_masked, cmap = 'RdBu', vmin = -rg_, vmax = rg_)
        plt.colorbar()
        plt.tight_layout()
        if not save is None:
            plt.savefig(save.split('.')[-1] + '.png')
            plt.close()
        else:
            plt.show()
    return diff_masked, binary_erosion_mask


def make_difference_v2(projection2, projection1):
    '''
    Provides an alternative approach to making a difference map. Instead of attempting to align the two T2 projection maps and then subtracting them (as done above), this function finds the spatially closest pixel in timepoint 1 for each pixel in timepoint 2, and then assigns the T2 delta to the timepoint 2 pixel location. This results in a difference map that is non-zero for each non-zero pixel in timepoint 2.  
    
    '''
    
    projection1, projection2 = align_projections(projection1,projection2)
    
    num_pix = np.sum(projection2!=0)
    where_projection2 = np.where(projection2)
    where_projection1 = np.where(projection1)

    diff = np.zeros_like(projection2)
    for i in range(num_pix):
        t2_projection2 = projection2[where_projection2[0][i], where_projection2[1][i]]

        distances_r = where_projection1[0] - where_projection2[0][i]
        distances_c = where_projection1[1] - where_projection2[1][i]
        distances = np.sqrt(distances_r**2 + distances_c**2)
#         index = np.argmin(distances)
#         t2_projection1 = projection1[where_projection1[0][index], where_projection1[1][index]]
        indices = np.argsort(distances)[0:10]
        t2_projection1 = np.mean(projection1[where_projection1[0][indices], where_projection1[1][indices]])
        
        change = t2_projection2 - t2_projection1
        diff[where_projection2[0][i], where_projection2[1][i]] = change
    
    kernel_size = 30
    middle_r = (kernel_size/2)-.5
    middle_c = (kernel_size/2)-.5
    kernel = np.ones((kernel_size,kernel_size))
    for i in (range(kernel_size)):
        for j in (range(kernel_size)):
            if np.linalg.norm([(i-middle_r),(j-middle_c)]) > 4:
                kernel[i,j]=0
                
    mask = eroded_and_mask(diff!=0,diff!=0, kernel)
    diff_masked = diff*mask
    return diff_masked, mask