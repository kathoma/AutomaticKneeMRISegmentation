import numpy as np
import pickle
import matplotlib.pyplot as plt
from loss_functions import dice_loss_test_volume
from difference_map_utils import make_difference
from cluster_utils import threshold_diffmaps, strip_empty_lines, resize

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



def make_projection_proportional(projection_dict):
    '''
    input:
    projection_dict['t2_projection'] = 2D numpy array with T2 map projection
    
    projection_dict['row_distance'] =  (float) physical distaince (in units of mm) for the medial-lateral width of the knee joint's cartilage
    
    projection_dict['column_distance'] (float) physical distance (in units of mm) for the anterior-posterior width of the knee joint's cartilage
    
    output:
    a resized version of the projection (2D numpy array) such that each pixel represents 1 mm^2
    '''
    projection = projection_dict['t2_projection']
    
    projection = resize(projection, (np.round(projection_dict['row_distance']), np.round(projection_dict['column_distance'])))
    
    return projection




def calculate_lesion_area(projection1_dict,
                          projection2_dict, 
                          value_threshold=None,
                          sigma_multiple=None,
                          area_value_threshold = None,
                          area_fraction_threshold = None,
                          area_percentile_threshold=None,
                          display=False):
    '''
    projection1_dict (dict): dictionary containing cartilage 2D projection from timepoint 1 and metadata for resizing it
    
    projection2_dict (dict): dictionary containing cartilage 2D projection from timepoint 2 and metadata for resizing it
    
    value_threshold (float): A threshold in units of miliseconds by which each cartilage T2 value will be compared to determine if it's an outlier pixel. If a value is given, sigma_multiple will be ignored
    
    sigma_multiple (float): A threshold in units of standard deviations by which each cartilage T2 value will be compared to determine if it's an outlier pixel. The standard deviation is calculated across all cartilage pixels in the array.
    
    area_value_threshold (float): A threshold in units of mm^2 by which each cluster of outlier pixels will be compared to determine if that cluster is big enough to be considered a lesion. If a value is given, area_fraction_threshold and area_percentile_threshold are ignored. 
    
    area_fraction_threshold (float): A threshold in the range (0,1) by which each cluster of outlier pixels will be compared to determine if that cluster is big enough to be considered a lesion. E.g. if area_fraction_threshold = 0.01, a cluster must occupy 1% of the full cartilage plate in order to be considered a lesion. If a value is given, area_percentile_threshold is ignored. 
    
    area_percentile_threshold (float): A threshold in the range (0,100) by which each cluster of outlier pixels will be compared to determine if that cluster is big enough to be considered a lesion. E.g. if area_percentile_threshold = 80, clusters that are in the top 20 percent of size for this subject will be kept and all other clusters will be discarded. 
    
    display (bool): If True, the lesions are plotted on top of an outline of the projected cartilage plate. 
    
    '''
    
    projection1 = make_projection_proportional(projection1_dict)
    projection2 = make_projection_proportional(projection2_dict)

    last_col = np.min([projection1.shape[1],projection2.shape[1]])
    projection1 = projection1[:,0:last_col-1]
    projection2 = projection2[:,0:last_col-1]

#         projection1 = np.load(timepoint1[i])
#         projection2 = np.load(timepoint2[i])
    dif, mask = make_difference(projection2,projection1)

    dif = strip_empty_lines(dif)
    mask = strip_empty_lines(mask)

    dmap,m = threshold_diffmaps(dif, 
                                mask, 
                                value_threshold=value_threshold,
                                sigma_multiple = sigma_multiple,
                                one_sided = 1,
                                area_value_threshold = area_value_threshold,
                                area_fraction_threshold = area_fraction_threshold,
                                area_percentile_threshold = area_percentile_threshold,#80, 
                                plot = False)

    lesion_percent = np.sum(dmap!=0)/np.sum(m)

    toshow = np.copy(m)
    toshow = (toshow>.5)*1
    toshow[dmap!=0]=2
    if display:
        plt.imshow(toshow)
        plt.title("Lesion Percentage:" + str(lesion_percent))
        plt.show()

    return dmap, lesion_percent

     
def calculate_group_lesion_area(timepoint1, 
                          timepoint2,
                          value_threshold=None,
                          sigma_multiple=None,
                          area_value_threshold = None,
                          area_fraction_threshold = None,
                          area_percentile_threshold=None,
                          display=False,
                          save_path_list=None):
    '''
    timepoint1 (list of strings): list of full file paths to dictionaries containing cartilage 2D projection from timepoint 1 and metadata for resizing it
    
    timepoint1 (list of strings): list of full file paths to dictionaries containing cartilage 2D projection from timepoint 2 and metadata for resizing it
    
    value_threshold (float): A threshold in units of miliseconds by which each cartilage T2 value will be compared to determine if it's an outlier pixel. If a value is given, sigma_multiple will be ignored
    
    sigma_multiple (float): A threshold in units of standard deviations by which each cartilage T2 value will be compared to determine if it's an outlier pixel. The standard deviation is calculated across all cartilage pixels in the array.
    
    area_value_threshold (float): A threshold in units of mm^2 by which each cluster of outlier pixels will be compared to determine if that cluster is big enough to be considered a lesion. If a value is given, area_fraction_threshold and area_percentile_threshold are ignored. 
    
    area_fraction_threshold (float): A threshold in the range (0,1) by which each cluster of outlier pixels will be compared to determine if that cluster is big enough to be considered a lesion. E.g. if area_fraction_threshold = 0.01, a cluster must occupy 1% of the full cartilage plate in order to be considered a lesion. If a value is given, area_percentile_threshold is ignored. 
    
    area_percentile_threshold (float): A threshold in the range (0,100) by which each cluster of outlier pixels will be compared to determine if that cluster is big enough to be considered a lesion. E.g. if area_percentile_threshold = 80, clusters that are in the top 20 percent of size for this subject will be kept and all other clusters will be discarded. 
    
    display (bool): If True, the lesions are plotted on top of an outline of the projected cartilage plate. 
    
    save_path_list (list of string): list of full file paths where the lesion maps should be saved as numpy arrays
    
    '''
    
    assert len(timepoint1)==len(timepoint2), "timepoint1 and timepoint2 need to be same length"
    if save_path_list:
        assert len(timepoint1) == len(save_path_list), "need exactly one file path for each image pair"
    
    lesion_percent_list = []
    
    
    for i in range(len(timepoint1)):

        with open(timepoint1[i], 'rb') as handle:
            projection1_dict = pickle.load(handle)
            
        with open(timepoint2[i], 'rb') as handle:
            projection2_dict = pickle.load(handle)
            
        dmap,lesion_percent = calculate_lesion_area(projection1_dict,
                                                  projection2_dict, 
                                                  value_threshold=value_threshold,
                                                  sigma_multiple=sigma_multiple,
                                                  area_value_threshold = area_value_threshold,
                                                  area_fraction_threshold = area_fraction_threshold,
                                                  area_percentile_threshold=area_percentile_threshold,
                                                  display=display)
        
        lesion_percent_list.append(lesion_percent)
        
        if save_path_list:
            np.save(save_path_list[i], dmap)
            
    return lesion_percent_list 


def calculate_lesion_dice_scores(source1_timepoint1,
                                 source1_timepoint2,
                                 source2_timepoint1,
                                 source2_timepoint2,
                                 value_threshold=None,
                                 sigma_multiple=None,
                                 area_value_threshold = None,
                                 area_fraction_threshold = None,
                                 area_percentile_threshold=None):
    
    '''
    Calculates the agreement in 2D lesion localization derived from two different segmentations of the same image
    
    input:
    source1_timepoint1 (list of strings): full file path to projection dictionaries for timepoint 1 of segmentation source 1
    source1_timepoint2 (list of strings): full file path to projection dictionaries for timepoint 2 of segmentation source 1
    source2_timepoint1 (list of strings): full file path to projection dictionaries for timepoint 1 of segmentation source 2
    source2_timepoint2 (list of strings): full file path to projection dictionaries for timepoint 2 of segmentation source 2
    
    output:
    list of dice scores. The ith dice score in this list expresses the agreement in the lesions identified from the pair (source1_timepoint1[i], source1_timepoint2[i]) and the lesions identified from the pair (source2_timepoint1[i], source2_timepoint2[i])
    
    
    '''
    
    assert len(source1_timepoint1)==len(source1_timepoint2)
    assert len(source2_timepoint1)==len(source2_timepoint2)
    assert len(source1_timepoint1)==len(source2_timepoint1)

    dice_list = []
    for i in range(len(source1_timepoint1)):

        with open(source1_timepoint1[i], 'rb') as handle:
            proj_dict_s1t1 = pickle.load(handle)

        with open(source1_timepoint2[i], 'rb') as handle:
            proj_dict_s1t2 = pickle.load(handle)


        lesion_s1, _ = calculate_lesion_area(proj_dict_s1t1,
                                              proj_dict_s1t2, 
                                              value_threshold=value_threshold,
                                              sigma_multiple=sigma_multiple,
                                              area_value_threshold = area_value_threshold,
                                              area_fraction_threshold = area_fraction_threshold,
                                              area_percentile_threshold=area_percentile_threshold,
                                              display=False)

        with open(source2_timepoint1[i], 'rb') as handle:
            proj_dict_s2t1 = pickle.load(handle)

        with open(source2_timepoint2[i], 'rb') as handle:
            proj_dict_s2t2 = pickle.load(handle)

        lesion_s2, _ = calculate_lesion_area(proj_dict_s2t1,
                                             proj_dict_s2t2, 
                                             value_threshold=value_threshold,
                                             sigma_multiple=sigma_multiple,
                                             area_value_threshold = area_value_threshold,
                                             area_fraction_threshold = area_fraction_threshold,
                                             area_percentile_threshold=area_percentile_threshold,
                                             display=False)

        lesion_s1, lesion_s2 = align_projections(lesion_s1, lesion_s2,True, False)

        dice = dice_loss_test_volume(lesion_s1!=0, lesion_s2!=0)
        dice_list.append(dice)

    return dice_list

      