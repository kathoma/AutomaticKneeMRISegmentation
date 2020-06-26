import scipy.ndimage as ndimage
import numpy as np

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

def eroded_and_mask(mata, matb, kernel = np.ones((2,2))):
    """
    Creatives mask for matrices *mata* and *matb* ( each 2D numpy.array)
    with positive elements.
    Erosion is applied first to each of matrix with *kernel* default(2x2)
    Then pointwise logical and is taken between these two masks.
    """
    assert mata.shape == matb.shape, "Shapes don't match"
    assert (mata>=0).all() and (matb>=0).all(), "Works only for positive matrices"
    ermask_a = ndimage.morphology.binary_erosion(1*(mata > 0),
                                    structure = kernel).astype(np.int)
    ermask_b = ndimage.morphology.binary_erosion(1*(matb > 0),
                                    structure = kernel).astype(np.int)
    return ermask_a*ermask_b

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
            
    
    binary_erosion_mask = eroded_and_mask(t1_proj, t2_proj)
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

