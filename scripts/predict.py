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
from cluster_utils import threshold_diffmaps, strip_empty_lines, resize
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

print("STARTING NOW")
input_dir = 'input'
vol_zip_list = np.sort([os.path.join(input_dir,i) for i in os.listdir(input_dir) if i[-4:]=='.zip'])

# Get model
model_weight_file = './model_weights/model_weights_quartileNormalization_echoAug.h5'
model = get_model(model_weight_file)

# Prepare CSV to write all results to
region_list = ['all', 'superficial', 'deep','L', 'M', 'LA', 'LC', 'LP', 'MA', 'MC', 'MP', 'SL', 'DL', 'SM', 'DM','SLA', 'SLC', 'SLP', 'SMA', 'SMC', 'SMP', 'DLA', 'DLC', 'DLP', 'DMA', 'DMC', 'DMP']
output_file = open('output/predictions.csv', 'w')
output_file.write('filename,'+','.join(region_list)+'\n')

total_time = 0
for zip_num, vol_zip in enumerate(vol_zip_list):
    print("Processing file %s..." % os.path.basename(vol_zip))
    time1 = time.time()
    new_dir_name = os.path.join('output', os.path.splitext(vol_zip)[0])
    print(new_dir_name)
    os.makedirs(new_dir_name, exist_ok=True)
    dicom_sub_dir = os.path.join(new_dir_name,"dicom")
    os.makedirs(dicom_sub_dir, exist_ok=True)
    
    with zipfile.ZipFile(vol_zip, 'r') as zip_ref:
        zip_ref.extractall(dicom_sub_dir)
    
    mese, times = assemble_4d_mese(dicom_sub_dir)
    
    # Whiten the echo of each slice that is closest to 10ms 
    mese_white = []
    for i,s in enumerate(mese):
        slice_times = times[i]-times[i][0]
        slice_10ms_idx = np.argmin(slice_times-.01)
        mese_white.append(whiten_img(s[slice_10ms_idx,:,:], normalization = 'quartile'))
    mese_white = np.stack(mese_white).squeeze()
    
    # Estimate segmentation
    seg_pred = model.predict(mese_white.reshape(-1,384,384,1), batch_size = 6)
    seg_pred = seg_pred.squeeze() # SAVE THIS
    
    # Calculate T2 Map
    t2 = fit_t2(mese, times, segmentation = seg_pred, n_jobs = 4, show_bad_pixels = False)
    
    # Refine the comparison segmentation by throwing out non-physiologic T2 values
    seg_pred, t2 = t2_threshold(seg_pred, t2, t2_low=0, t2_high=100)
    seg_pred, t2 = optimal_binarize(seg_pred, t2, prob_threshold=0.501, voxel_count_threshold=425)
        
    angular_bin = 5
    visualization, thickness_map, min_rho_map, max_rho_map, avg_vals_dict, R = projection(t2, 
                                                                                       thickness_div = 0.5, 
                                                                                       values_threshold = 100,
                                                                                       angular_bin = angular_bin, 
                                                                                       region_stat = 'mean',
                                                                                       fig = False)

    row_distance, column_distance = get_physical_dimensions(img_dir = dicom_sub_dir, 
                                                            t2_projection = visualization, 
                                                            projection_pixel_radius = R, 
                                                            angular_bin = angular_bin)
    
    t2_projection_dict = {}
    t2_projection_dict['t2_projection'] = visualization
    t2_projection_dict['thickness_map'] = thickness_map
    t2_projection_dict['row_distance'] = row_distance
    t2_projection_dict['column_distance'] = column_distance

    # Save the t2 image, segmentation, and projection results
    
    ## Save the 3D binary segmentation mask as a numpy array
    refined_seg_path = os.path.join(new_dir_name,"segmentation_mask.npy")
    np.save(refined_seg_path, seg_pred)
    
    ## Save the 3D binary segmentation mask as a folder of CSV files
    seg_sub_dir = os.path.join(new_dir_name,"segmentation_mask_csv")
    os.makedirs(seg_sub_dir, exist_ok=True)
    
    for i,s in enumerate(seg_pred):
        slice_path = os.path.join(seg_sub_dir,str(i).zfill(3)+".csv")
        np.savetxt(slice_path, s,delimiter=",", fmt='%10.5f')
                
    ## Save the 3D T2 image as a numpy array
    t2_img_path = os.path.join(new_dir_name,"t2.npy")
    np.save(t2_img_path, t2)
    
    ## Save the 3D T2 image as a folder of CSV files
    t2_sub_dir = os.path.join(new_dir_name,"t2_csv")
    os.makedirs(t2_sub_dir, exist_ok=True)
    
    for i,s in enumerate(t2):
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
    
    # Write to master CSV
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
print(os.listdir('./output'))
print("Processing finished.")