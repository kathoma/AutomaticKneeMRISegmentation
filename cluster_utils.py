import skimage
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

def threshold_matrix(mat, thr = 1):
    "Makes two - sided thresholding of 2D matrix *mat* based on threshold *thr* (defulat=1)"
    matthresh = mat.copy()
    matthresh[np.where((thr > mat) & (mat > 0))] = 0
    matthresh[np.where((-thr < mat) & (mat < 0))] = 0
    return matthresh

def threshold_matrix_twosided(mat, upper = 1, lower = -1):
    "Makes two - sided thresholding of 2D matrix *mat* based on threshold *thr* (defulat=1)"
    matthresh = mat.copy()
    matthresh[np.where((upper > mat) & (mat > 0))] = 0
    matthresh[np.where((lower < mat) & (mat < 0))] = 0
    return matthresh

def make_mask_from_coords_list(mat, coords_list, val = 1):
    """
    Puts *val* (int) on *mat* (2D numpy.array) given coordinates
    that are list of lists with pairs of values
    """
    for coords in coords_list:
        for (x,y) in coords:
            mat[x,y] = val
    return mat

def strip_empty_lines(mat):
    newmat = mat[np.sum(mat,axis = 1)!=0,:]
    newmat = newmat[:,np.sum(newmat, axis = 0)!=0]
#     newmat = mat[~np.all(mat == 0, axis=1)].T
#     newmat = newmat[~np.all(newmat == 0, axis=1)].T
    return newmat

def resize(mat, target_size = (22,44)):
    resized = skimage.transform.resize(mat, 
                                         target_size, 
                                         order=1, 
                                         mode='edge', 
                                         clip=True, 
                                         preserve_range=True, 
                                         anti_aliasing=True, 
                                         anti_aliasing_sigma=None)
    
#     cv2.resize(mat, dsize = , interpolation = cv2.INTER_NEAREST )
    return resized

def get_values_std(list_of_diffs, mode = 0):
    '''
    Iterates over difference values and returns standard deviation.
    IN:
        list_of_diffs - list of files with difference and masks (check *make_difference* for details)
        model - 0 computes STD of all values, 1 return mean of individual STDs (default: 0)
    OUT:
        sigma - standard deviation of values from difference maps.
    Example:
        > import glob
        > list_diffs = glob.glob(difference_maps_folder + '*.npz')
        > get_values_std(list_diffs, 1)
    '''
    list_diff_values = []
    list_diff_stds   = []
    for fname in list_of_diffs:
        loader = np.load(fname)
        diffmap = loader['diff']
        mask    = loader['mask']
        list_diff_values.extend(list(diffmap[np.where(mask>0)]))
        list_diff_stds.append(diffmap[np.where(mask>0)].std())
    if mode == 0:
        sigma = np.std(list_diff_values)
    else:
        sigma2 = np.mean(list_diff_stds)
    return sigma

def threshold_diffmaps(diffmap, 
                       mask, 
                       value_threshold,
                       sigma_multiple = None, 
                       one_sided = 0,
                       area_value_threshold = None,
                       area_fraction_threshold = None,
                       area_percentile_threshold = None,#80, 
                       plot = False):
    '''
    Performs clustering in two steps:
      1) Based on values - must be below, above sigma.
      2) Based on size of pixel area.
    IN:
        diffmap - matrix with difference map (n, m)
        mask - binary matrix with cartilage region annotation as 1 (n, m)
        sigma_threshold - value of sigma; if None (default) sigma is calculated based
               from diffmap and +/- 1.5 * sigma threshold is taken
               (values above sigma_threshold survive)
        one_sided - thresholding from positive (+1) or negative (-1) sides only,
               (default both: 0)
        area_threshold - percentile of cluste areas to survive, eg. 80 (default)
               is 80-th percentile
        plot - if string is given it saves plot with steps of thresholding to this path
               (default is None - doesn't plot)
    OUT:
        diff_thresh_size - difference map after thresholding
    '''
    sigma_in = diffmap[np.where(mask>0)].std()
    mean_in = diffmap[np.where(mask>0)].mean()
    st_diffmap = strip_empty_lines(diffmap + mask)
    st_mask = strip_empty_lines(mask)
    
    # RECORD THE NUMBER OF ROWS (IE SLICES) THAT CONTAIN CARTILAGE HERE
    
#     st_diffmap = resize(st_diffmap, target_size = (st_diffmap.shape[0]*100, st_diffmap.shape[1]*100))
#     st_mask = resize(st_mask, target_size = (st_mask.shape[0]*100, st_mask.shape[1]*100))
    st_mask = (st_mask>.5)*1

    diffmap = st_diffmap - st_mask
    
    
    # Step (1) - Thresholding based on sigma
    val_threshold = value_threshold if value_threshold else (sigma_multiple * sigma_in)
    if one_sided != 0:
        if one_sided > 0:
            diff_thresh = threshold_matrix_twosided(diffmap, \
                                                    mean_in + val_threshold, -np.inf)
        else:
            diff_thresh = threshold_matrix_twosided(diffmap, \
                                                    np.inf, mean_in - val_threshold)
    else:
        diff_thresh = threshold_matrix_twosided(diffmap, \
                                                mean_in + val_threshold, mean_in - val_threshold)
    # Determining area sizes
    arrlabeled = skimage.measure.label(diff_thresh != 0)
    regions = skimage.measure.regionprops(arrlabeled)
    areas = [region['area'] for region in regions]
    
    
    # Step (2) - Thresholding based on area
    if area_value_threshold:
        cutoff_size = area_value_threshold
    elif area_fraction_threshold:
        cutoff_size = np.sum(st_mask)*area_fraction_threshold
    elif area_percentile_threshold:
        cutoff_size = ss.scoreatpercentile(areas, area_percentile_threshold)
    coords_list = [reg.coords for reg, ar in zip(regions, areas) if ar > cutoff_size]
    mask_sized = make_mask_from_coords_list(np.zeros(diff_thresh.shape), coords_list) 
    diff_thresh_size = diff_thresh * mask_sized
    
    if plot:
        plt.figure()
        plt.subplot(311)
        plt.hist(areas, bins=10)
        plt.title('Histogram of areas')
        for sc in [80, 85, 90, 95]:
            plt.axvline(x = ss.scoreatpercentile(areas, sc), color='r')
        plt.xlabel('Areas size (px)')
        plt.subplot(312)
        plt.imshow(arrlabeled)
        plt.title('Pixel clusters')
        plt.subplot(313)
        rg_ = np.max([abs(np.max(diff_thresh_size)), abs(np.min(diff_thresh_size))])
        plt.imshow(diff_thresh_size * mask_sized, cmap='RdBu', vmin=-rg_, vmax=rg_)
        plt.colorbar()
        plt.imshow(st_mask, cmap='binary', alpha=0.2)
        plt.title(r"Difference values after whole clustering (80% area cut-off)")
        plt.tight_layout()
#         plt.savefig(plot)
        plt.show()
    return diff_thresh_size, st_mask

