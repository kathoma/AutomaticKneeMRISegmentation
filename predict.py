from __future__ import print_function, division
import sys
sys.path.insert(0, 'lib')
import numpy as np
import random
import pydicom
import os
import matplotlib.pyplot as plt
import pickle
import math
import pydicom
from shutil import copyfile
import nibabel as nib
import scipy.ndimage as ndimage
from scipy.stats import pearsonr, spearmanr

from utils import make_giant_mat, make_dictionary, make_echo_dict
from difference_map_utils import make_difference
from cluster_utils import threshold_diffmaps, strip_empty_lines
from lesion_utils import *
from inference_utils import run_inference
from make_inference_csv import *
from compare_segmentations import get_dice_scores, get_jaccard_indices, compare_segmentation_masks, compare_region_means, compare_region_changes
from loss_functions import coefficient_of_variation
from figure_utils import plot_mean_val_comparisons

import zipfile
from calculate_t2 import fit_t2
from segmentation_refinement import *
from projection import *
from inference_utils import *
import time
import shutil
from skimage.transform import resize

print("STARTING NOW")
input_dir = '/workspace/input'
vol_zip_list = np.sort([os.path.join(input_dir,i) for i in os.listdir(input_dir) if i[-4:]=='.zip'])

# Get model
model_weight_file = 'workspace/model_weights/model_weights_quartileNormalization_echoAug.h5'
model = get_model(model_weight_file)

# Prepare CSV to write all results to
region_list = ['all', 'superficial', 'deep','L', 'M', 'LA', 'LC', 'LP', 'MA', 'MC', 'MP', 'SL', 'DL', 'SM', 'DM','SLA', 'SLC', 'SLP', 'SMA', 'SMC', 'SMP', 'DLA', 'DLC', 'DLP', 'DMA', 'DMC', 'DMP']
output_file = open('workspace/output/results_summary.csv', 'w')
output_file.write('filename,'+','.join(region_list)+'\n')

# Define functions that will help us find the dicom files in the input zip directories
def get_slices(scan_dir):
    '''
    scan_dir = path to folder containing dicoms such that each dicom represents one slice of the image volume
    '''
    list_of_slices = glob.glob("{}/**".format(scan_dir),
                               recursive=True)
    return list(filter(is_dicom, list_of_slices))

def is_dicom(fullpath):
    if os.path.isdir(fullpath):
        return False
    _, path = os.path.split(fullpath)
    
    path = path.lower()
    if path[-4:] == ".dcm":
        return True
    if not "." in path:
        return True
    return False


# Process input zip directories
total_time = 0
for zip_num, vol_zip in enumerate(vol_zip_list):
    print("Processing file %s..." % os.path.basename(vol_zip))
    time1 = time.time()
    new_dir_name = os.path.join('/workspace/output', os.path.splitext(os.path.basename(vol_zip))[0])
    os.makedirs(new_dir_name, exist_ok=True)
    dicom_sub_dir = os.path.join(new_dir_name,"dicom")
    raw_extract_dir = os.path.join(new_dir_name,"raw_extract")
    os.makedirs(dicom_sub_dir, exist_ok=True)
    os.makedirs(raw_extract_dir, exist_ok=True)
    
    # Unzip the image volume. If the zip file contains an inner directory, move the files out of it. 
    with zipfile.ZipFile(vol_zip, 'r') as zip_ref:
        zip_ref.extractall(raw_extract_dir)
    
    slice_path_list = get_slices(raw_extract_dir)
    for s in slice_path_list:
        shutil.copy(s,os.path.join(dicom_sub_dir, os.path.basename(s)))
    shutil.rmtree(raw_extract_dir)

    # Create a MESE numpy array
    mese, times = assemble_4d_mese_v2(dicom_sub_dir)
    
    # If the slices are not 384x384, resize them (the model was trained on 384x384 images from OAI)
    original_shape = None
    if ((mese.shape[-1] != 384) or (mese.shape[-1] != 384)):
        original_shape = mese.shape
        mese_resized = np.zeros((mese.shape[0], mese.shape[1], 384,384))
        for s in range(mese.shape[0]):
            for echo in range(mese.shape[1]):
                mese_resized[s,echo,:,:] = resize(mese[s,echo,:,:], (384, 384),anti_aliasing=True)
        mese = mese_resized  
    
    # Whiten the echo of each slice that is closest to 20ms 
    mese_white = []
    for i,s in enumerate(mese):
        if times[i][0] is not None:
            slice_times = times[i]
        else:
            times[i][0]=times[i][1]-(times[i][2]-times[i][1])
            slice_times = times[i]
        slice_20ms_idx = np.argmin(slice_times-.02)
        mese_white.append(whiten_img(s[slice_20ms_idx,:,:], normalization = 'quartile'))
    mese_white = np.stack(mese_white).squeeze()
    
    # Estimate segmentation
    seg_pred = model.predict(mese_white.reshape(-1,384,384,1), batch_size = 6)
    seg_pred = seg_pred.squeeze() # SAVE THIS
    
    # Calculate T2 Map
    t2 = fit_t2(mese, times, segmentation = seg_pred, n_jobs = 4, show_bad_pixels = False)

    # Refine the comparison segmentation by throwing out non-physiologic T2 values
    seg_pred_refined, t2_refined = t2_threshold(seg_pred, t2, t2_low=0, t2_high=100)
    seg_pred_refined, t2_refined = optimal_binarize(seg_pred_refined, t2_refined, prob_threshold=0.501,voxel_count_threshold=425)
    
        
    angular_bin = 5
    visualization, thickness_map, min_rho_map, max_rho_map, avg_vals_dict, R = projection(t2_refined, 
                                                                                       thickness_div = 0.5, 
                                                                                       values_threshold = 100,
                                                                                       angular_bin = angular_bin, 
                                                                                       region_stat = 'mean',
                                                                                       fig = False)

    row_distance, column_distance = get_physical_dimensions(img_dir = dicom_sub_dir, 
                                                            t2_projection = visualization, 
                                                            projection_pixel_radius = R, 
                                                            angular_bin = angular_bin)
    
    # Resize the output to the size of the original input image
    if original_shape is not None:
        seg_pred_resized = np.zeros((original_shape[0], original_shape[2],original_shape[3]))
        t2_resized = np.zeros((original_shape[0], original_shape[2],original_shape[3]))
        seg_pred_refined_resized = np.zeros((original_shape[0], original_shape[2],original_shape[3]))
        t2_refined_resized = np.zeros((original_shape[0], original_shape[2],original_shape[3]))
        for s in range(seg_pred.shape[0]):
            seg_pred_resized[s,:,:] = resize(seg_pred[s,:,:](original_shape[2], original_shape[3]), anti_aliasing=True, preserve_range=True)
            t2_resized[s,:,:] = resize(t2[s,:,:], (original_shape[2],original_shape[3]),anti_aliasing=True,preserve_range=True)
            seg_pred_refined_resized[s,:,:] = resize(seg_pred_refined[s,:,:], (original_shape[2], original_shape[3]), anti_aliasing=True, preserve_range=True)
            t2_refined_resized[s,:,:] = resize(t2_refined[s,:,:], (original_shape[2], original_shape[3]), anti_aliasing=True, preserve_range=True)

        seg_pred = 1*(seg_pred_resized>.501)
        seg_pred_refined = 1*(seg_pred_refined_resized>.501)
        t2 = t2_resized
        t2_refined = t2_refined_resized

    # Save the t2 image, segmentation, and projection results
        
    ## Save the 3D binary segmentation mask as a numpy array
    seg_path = os.path.join(new_dir_name,"segmentation_mask.npy")
    np.save(seg_path, seg_pred)
    
    refined_seg_path = os.path.join(new_dir_name,"segmentation_mask_refined.npy")
    np.save(refined_seg_path, seg_pred_refined)
    
    ## Save the 3D binary segmentation mask as a folder of CSV files
    seg_sub_dir = os.path.join(new_dir_name,"segmentation_mask_csv")
    os.makedirs(seg_sub_dir, exist_ok=True)
    
    for i,s in enumerate(seg_pred):
        slice_path = os.path.join(seg_sub_dir,str(i).zfill(3)+".csv")
        np.savetxt(slice_path, s,delimiter=",", fmt='%10.5f')
        
    seg_sub_dir = os.path.join(new_dir_name,"segmentation_mask_csv_refined")
    os.makedirs(seg_sub_dir, exist_ok=True)
    
    for i,s in enumerate(seg_pred_refined):
        slice_path = os.path.join(seg_sub_dir,str(i).zfill(3)+".csv")
        np.savetxt(slice_path, s,delimiter=",", fmt='%10.5f')
                
    ## Save the 3D T2 image as a numpy array
    t2_img_path = os.path.join(new_dir_name,"t2.npy")
    np.save(t2_img_path, t2)
    
    t2_img_path_refined = os.path.join(new_dir_name,"t2_refined.npy")
    np.save(t2_img_path_refined, t2_refined)
    
    ## Save the 3D T2 image as a folder of CSV files
    t2_sub_dir = os.path.join(new_dir_name,"t2_csv")
    os.makedirs(t2_sub_dir, exist_ok=True)
    
    for i,s in enumerate(t2):
        slice_path = os.path.join(t2_sub_dir,str(i).zfill(3)+".csv")
        np.savetxt(slice_path, s,delimiter=",", fmt='%10.5f')
        
    t2_sub_dir = os.path.join(new_dir_name,"t2_csv_refined")
    os.makedirs(t2_sub_dir, exist_ok=True)
    
    for i,s in enumerate(t2_refined):
        slice_path = os.path.join(t2_sub_dir,str(i).zfill(3)+".csv")
        np.savetxt(slice_path, s,delimiter=",", fmt='%10.5f')
    
    ## Save the 2D projection of the T2 map as a numpy array
    t2_projection_path = os.path.join(new_dir_name,"t2_projection.npy")
    np.save(t2_projection_path, visualization)
    
    ## Save the 2D projection of the T2 map as a csv
    t2_projection_csv_path = os.path.join(new_dir_name,"t2_projection.csv")
    np.savetxt(t2_projection_csv_path, visualization,delimiter=",", fmt='%10.5f')
    
    ## Save the 2D projection thickness map as a numpy array
    thickness_projection_path = os.path.join(new_dir_name,"thickness_projection.npy")
    np.save(thickness_projection_path, thickness_map)
    
    ## Save the 2D projection thickness map as a csv
    thickness_projection_csv_path = os.path.join(new_dir_name,"thickness_projection.csv")
    np.savetxt(thickness_projection_csv_path, thickness_map,delimiter=",", fmt='%10.5f')
    
    ## Save the physical dimensions of the 2D projections as a json
    projection_dimensions_dict = {}
    projection_dimensions_dict['row_distance(mm)'] = row_distance
    projection_dimensions_dict['column_distance(mm)'] = column_distance
    projection_dimensions_dict_path = os.path.join(new_dir_name,"projection_dimensions.json")
    with open(projection_dimensions_dict_path, 'w') as fp:
        json.dump(projection_dimensions_dict, fp)
        
    ## Save the region average T2 dictionary as a json
    t2_region_json_path = os.path.join(new_dir_name,"region_mean_t2.json")
    with open(t2_region_json_path, 'w') as fp:
        json.dump(avg_vals_dict, fp)   
    
    # Record the average regional T2 values for this image to a summary CSV file where we're recording these metrics for all input images
    output_file.write('%s,' % os.path.basename(vol_zip))
    for r in region_list:
        if r == 'DMP':
            output_file.write('%d' % avg_vals_dict[r])
        else:
            output_file.write('%d,' % avg_vals_dict[r])

    output_file.write('\n')
    
    time2 = time.time()
    total_time = total_time + (time2-time1)
    avg_pace = total_time / (zip_num+1)
    files_remaining = len(vol_zip_list) - zip_num
    print("Estimated time remaining for all images (minutes):",np.round(files_remaining*avg_pace/60,decimals=1)) 
    
output_file.close()
print()
print("Processing finished. Find results in the 'output' folder:")
print()