import numpy as np
import pydicom
import os
import nibabel as nib
import pandas as pd
import json
import argparse
import pickle
import time

from calculate_t2 import *
from segmentation_refinement import *
from projection import *
from utils import whiten_img
from models import *
from cluster_utils import strip_empty_lines



def get_model(model_weight_file):
    '''
    model_weight_file = h5 model weight file for keras unet
    '''
    model = unet_2d_model((384, 384, 1))
    model.load_weights(model_weight_file)
    return model

def get_expert_seg_numpy(path_to_nifti):
    '''
    loads a nifti segmentation 3D volume supplied by user and converts to numpy
    
    path_to_nifti (str): full path to nifti file
    '''
    mask = nib.load(path_to_nifti)
    mask = mask.get_fdata()
    mask = np.transpose(mask,(2,1,0))
    return mask

def assemble_4d_mese(img_dir):
    '''
    assembles a 4D multi echo spin echo
    
    inputs:
        img_dir (str): path to a directory that contains all the echoes for all the slices for one MRI volume
    
    outputs:
        4D MESE image as a np array of shape (slices, echoes, height, width)

        dict specifying the echo times for each slice. 
            keys are ints specifying slice index. 
            vals are lists of floats specifying echo times 
    
    '''
    file_list = np.sort(os.listdir(img_dir))
    instance_arr = np.empty((len(file_list),2))
    
    # Find how many slices and echos are in this volume
    for i,file in enumerate(file_list):
        try:
            dcm = pydicom.read_file(os.path.join(img_dir,file))
        except:
            print("non-dicom file encountered:", file)
            continue
        
        instance_arr[i,0] = dcm.InstanceNumber
        instance_arr[i,1] = dcm.EchoNumbers
    
    num_echoes = int(len(np.unique(instance_arr[:,1])))
    num_slices = int(len(instance_arr)/num_echoes)
    height = dcm.pixel_array.shape[0]
    width = dcm.pixel_array.shape[1]
    vol = np.empty((int(num_slices), int(num_echoes), height, width))
    times = {}
    for i in range(num_slices):
        times[i]=np.array([None]*num_echoes)
        
    
    for i,file in enumerate(file_list):
        try:
            dcm = pydicom.read_file(os.path.join(img_dir,file))
            
        except:
            pass
        
        slice_idx = (dcm.InstanceNumber-1)%num_slices
        echo_idx = np.where(int(dcm.EchoNumbers)==np.unique(instance_arr[:,1]))[0][0]
        
        vol[int(slice_idx), echo_idx,:,:] = dcm.pixel_array
        times[int(slice_idx)][echo_idx] = float(dcm.AcquisitionTime)
    
    return vol, times

def get_metadata(img_dir):
    '''
    inputs:
    img_dir: string specifying the full path to the folder that contains all the dicom files for the image
    
    outputs:
    specified metadata from the first slice's dicom file in the image directory. 

    '''
    f = np.sort(os.listdir(img_dir))[0]
    f = os.path.join(img_dir,f)
    dcm = pydicom.read_file(f)
    slice_thickness = dcm.SliceThickness
    slice_spacing = dcm.SpacingBetweenSlices
    pixel_spacing = dcm.PixelSpacing[0]
    manufacturer = dcm.Manufacturer
    
    return slice_thickness, slice_spacing, pixel_spacing, manufacturer
    
    
def get_physical_dimensions(img_dir, t2_projection, projection_pixel_radius,angular_bin = 5):
    # Find the physical area covered by the cartilage plate
    '''
    inputs:
    img_dir: string specifying the full path to the folder that contains all the dicom files for the image
    
    t2_projection: 2D numpy array containing the 2D projection of the T2 map for the image
    
    projection_pixel_radius: the radius (in units of pixels) for the best-fit circle calculated during the 2D projection step
    
    outputs:
    row_distance (float): physical distaince (in units of mm) for the medial-lateral width of the knee joint's cartilage
    
    column_distance (float): physical distance (in units of mm) for the anterior-posterior width of the knee joint's cartilage
    
    '''
    ### Find the slice spacing and slice thickness in order to calculate the medial-lateral distance of the cartilage plate
    slice_thickness, slice_spacing, pixel_spacing, manufacturer = get_metadata(img_dir)
    num_slices = t2_projection.shape[0]
    row_distance = (num_slices * slice_thickness) + ((num_slices - 1) * slice_spacing)
#     temp = np.copy(t2_projection)
#     for i in np.argwhere(np.isnan(temp)): 
#             temp[tuple(i)]=0

#     temp = strip_empty_lines(temp)
#     num_cartilage_slices = temp.shape[0]
#     row_distance = (num_cartilage_slices * slice_thickness) + ((num_cartilage_slices - 1) * slice_spacing)

    ### Calculate the anterior-posterior distance of the cartilage plate
    radius = projection_pixel_radius*pixel_spacing
    column_distance = 2*radius*np.pi

#     full_circumference = 2*radius*np.pi
#     circle_percent = angular_bin*temp.shape[1]/360
#     column_distance = full_circumference*circle_percent
    
    return row_distance, column_distance

    

def process_expert_segmentations(expert_file_array):

    for i in range(len(expert_file_array)):
        
        img_dir = np.array(expert_file_array.img_dir)[i]
        seg_path = np.array(expert_file_array.seg_path)[i]
        print(seg_path)
        
        refined_seg_path = np.array(expert_file_array.refined_seg_path)[i]
        t2_img_path = np.array(expert_file_array.t2_img_path)[i]
        t2_projected_path = np.array(expert_file_array.t2_projected_path)[i]
        t2_region_json_path = np.array(expert_file_array.t2_region_json_path)[i]
        
        if seg_path[-6:]=='nii.gz':
            
            # Retrieve expert's initial segmentation
            seg_expert = get_expert_seg_numpy(seg_path)
            
            # Save as numpy array in the same folder as where the nifti files are
            seg_np_path = os.path.join(os.path.dirname(seg_path),seg_path[:-7])
            np.save(seg_np_path, seg_expert)
            
        elif seg_path[-3:]=='npy':
            seg_expert = np.load(seg_path)
            
        else:
            print("segmentation must be a nifti or numpy file:", seg_path)
            continue
            
        # Retrieve the corresponding MRI and echo times
        mese_expert, t2times_expert = assemble_4d_mese(img_dir)

        # Calculate T2 Map
        t2_expert = fit_t2(mese_expert, t2times_expert, segmentation = seg_expert, n_jobs = 4, show_bad_pixels = False)

        # Refine the comparison segmentation by throwing out non-physiologic T2 values
        seg_expert, t2_expert = t2_threshold(seg_expert, t2_expert, t2_low=0, t2_high=100)

        # Project the t2 map into 2D
        angular_bin = 5
        visualization, thickness_map, min_rho_map, max_rho_map, avg_vals_dict, R = projection(t2_expert, 
                                                                                               thickness_div = 0.5, 
                                                                                               values_threshold = 100,
                                                                                               angular_bin = angular_bin, 
                                                                                               region_stat = 'mean',
                                                                                               fig = False)
        
        row_distance, column_distance = get_physical_dimensions(img_dir = img_dir, 
                                                                t2_projection = visualization, 
                                                                projection_pixel_radius = R, 
                                                                angular_bin = angular_bin)
        
        t2_projection_dict = {}
        t2_projection_dict['t2_projection'] = visualization
        t2_projection_dict['row_distance'] = row_distance
        t2_projection_dict['column_distance'] = column_distance
        
        
        # Save the t2 image, segmentation, and projection results
        if not os.path.isdir(os.path.dirname(t2_img_path)):
            os.mkdir(os.path.dirname(t2_img_path))
        np.save(t2_img_path, t2_expert)
        
        if not os.path.isdir(os.path.dirname(refined_seg_path)):
            os.mkdir(os.path.dirname(refined_seg_path))
        np.save(refined_seg_path, seg_expert)
        
        if not os.path.isdir(os.path.dirname(t2_projected_path)):
            os.mkdir(os.path.dirname(t2_projected_path))
#         np.save(t2_projected_path, visualization)
        with open(t2_projected_path+'.pickle', 'wb') as handle:
            pickle.dump(t2_projection_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         with open(t2_projected_path, 'w') as fp:
#             json.dump(t2_projection_dict, fp)

        if not os.path.isdir(os.path.dirname(t2_region_json_path)):
            os.mkdir(os.path.dirname(t2_region_json_path))
            
        with open(t2_region_json_path, 'w') as fp:
            json.dump(avg_vals_dict, fp)
            
                      

def model_segmentation(file_array, model_weight_file, normalization = 'quartile'):
    
    for i in range(len(file_array)):
        
        img_dir = np.array(file_array.img_dir)[i]
        print(img_dir)
        seg_path = np.array(file_array.seg_path)[i]
        
        refined_seg_path = np.array(file_array.refined_seg_path)[i]
        t2_img_path = np.array(file_array.t2_img_path)[i]
        t2_projected_path = np.array(file_array.t2_projected_path)[i]
        t2_region_json_path = np.array(file_array.t2_region_json_path)[i]
        
        # Retrieve the corresponding MRI and echo times (slices, echoes, height, width)
        mese, t2times = assemble_4d_mese(img_dir) 
        
        # Whiten echo 1 of each slice
        mese_white = [whiten_img(s, normalization = normalization) for s in mese[:,1,:,:]]
        mese_white = np.stack(mese_white)
        
        # Get model
        model = get_model(model_weight_file)
        
        # Estimate segmentation
        time1 = time.time()
        seg_pred = model.predict(mese_white.reshape(-1,384,384,1), batch_size = 6)
        time2 = time.time()
        seg_pred = seg_pred.squeeze()
        if not os.path.isdir(os.path.dirname(seg_path)):
            os.mkdir(os.path.dirname(seg_path))
        np.save(seg_path, seg_pred)
        
        # Calculate T2 Map
        time3 = time.time()
        t2 = fit_t2(mese, t2times, segmentation = seg_pred, n_jobs = 4, show_bad_pixels = False)

        # Refine the comparison segmentation by throwing out non-physiologic T2 values
        seg_pred, t2 = t2_threshold(seg_pred, t2, t2_low=0, t2_high=100)
        seg_pred, t2 = optimal_binarize(seg_pred, t2, prob_threshold=0.501, voxel_count_threshold=425)
        time4 = time.time()
        
        print("Duration:", time2-time1, time4-time3, (time2-time1)+(time4-time3))
        # Project the t2 map into 2D
        angular_bin = 5
        visualization, thickness_map, min_rho_map, max_rho_map, avg_vals_dict, R = projection(t2, 
                                                                                           thickness_div = 0.5, 
                                                                                           values_threshold = 100,
                                                                                           angular_bin = angular_bin, 
                                                                                           region_stat = 'mean',
                                                                                           fig = False)
            
            
        row_distance, column_distance = get_physical_dimensions(img_dir = img_dir, 
                                                                        t2_projection = visualization, 
                                                                        projection_pixel_radius = R, 
                                                                        angular_bin = angular_bin)

        t2_projection_dict = {}
        t2_projection_dict['t2_projection'] = visualization
        t2_projection_dict['row_distance'] = row_distance
        t2_projection_dict['column_distance'] = column_distance
        
        
        # Save the t2 image, segmentation, and projection results
        if not os.path.isdir(os.path.dirname(t2_img_path)):
            os.mkdir(os.path.dirname(t2_img_path))
        np.save(t2_img_path, t2)
        
        if not os.path.isdir(os.path.dirname(refined_seg_path)):
            os.mkdir(os.path.dirname(refined_seg_path))
        np.save(refined_seg_path, seg_pred)
        
        if not os.path.isdir(os.path.dirname(t2_projected_path)):
            os.mkdir(os.path.dirname(t2_projected_path))
        with open(t2_projected_path+'.pickle', 'wb') as handle:
            pickle.dump(t2_projection_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if not os.path.isdir(os.path.dirname(t2_region_json_path)):
            os.mkdir(os.path.dirname(t2_region_json_path))
            
        with open(t2_region_json_path, 'w') as fp:
            json.dump(avg_vals_dict, fp)
            
            
            

def run_inference(expert_pd = None, 
                  to_segment_pd = None, 
                  model_weight_file = './model_weights_quartileNormalization_echoAug.h5'):
#'/data/kevin_data/checkpoints/checkpoint_weightsOnly_echo1_nomalization_quartile_dropOut0_augTrue_epoch17_trained_on_48_patients_valLoss0.8243473847938046.h5'):
    if expert_pd is not None:
        print("--------------------------------------------------")
        print("Using provided segmentations to analyze MESE MRIs")
        print("--------------------------------------------------")
        print()
        process_expert_segmentations(expert_pd)
    
    if to_segment_pd is not None:
        print("-----------------------------------------------------------------------------------")
        print("Automatically segmenting images and using those segmentations to analyze MESE MRIs")
        print("-----------------------------------------------------------------------------------")
        print()
        model_segmentation(to_segment_pd, model_weight_file = model_weight_file)
    
        























