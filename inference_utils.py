import numpy as np
import pydicom
import os
import nibabel as nib
import pandas as pd
import json
import argparse
from calculate_t2 import *
from segmentation_refinement import *
from projection import *
from utils import whiten_img
from models import *


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
        visualization, thickness_map, min_rho_map, max_rho_map, avg_vals_dict = projection(t2_expert, 
                                                                                           thickness_div = 0.5, 
                                                                                           values_threshold = 100,
                                                                                           angular_bin = 5, 
                                                                                           region_stat = 'mean',
                                                                                           fig = False)
        # Save the t2 image, segmentation, and projection results
        if not os.path.isdir(os.path.dirname(t2_img_path)):
            os.mkdir(os.path.dirname(t2_img_path))
        np.save(t2_img_path, t2_expert)
        
        if not os.path.isdir(os.path.dirname(refined_seg_path)):
            os.mkdir(os.path.dirname(refined_seg_path))
        np.save(refined_seg_path, seg_expert)
        
        if not os.path.isdir(os.path.dirname(t2_projected_path)):
            os.mkdir(os.path.dirname(t2_projected_path))
        np.save(t2_projected_path, visualization)

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
        seg_pred = model.predict(mese_white.reshape(-1,384,384,1), batch_size = 6)
        seg_pred = seg_pred.squeeze()
        if not os.path.isdir(os.path.dirname(seg_path)):
            os.mkdir(os.path.dirname(seg_path))
        np.save(seg_path, seg_pred)
        
        # Calculate T2 Map
        t2 = fit_t2(mese, t2times, segmentation = seg_pred, n_jobs = 4, show_bad_pixels = False)

        # Refine the comparison segmentation by throwing out non-physiologic T2 values
        seg_pred, t2 = t2_threshold(seg_pred, t2, t2_low=0, t2_high=100)
        seg_pred, t2 = optimal_binarize(seg_pred, t2, prob_threshold=0.501, voxel_count_threshold=425)
        
        # Project the t2 map into 2D
        visualization, thickness_map, min_rho_map, max_rho_map, avg_vals_dict = projection(t2, 
                                                                                           thickness_div = 0.5, 
                                                                                           values_threshold = 100,
                                                                                           angular_bin = 5, 
                                                                                           region_stat = 'mean',
                                                                                           fig = False)

        # Save the t2 image, segmentation, and projection results
        if not os.path.isdir(os.path.dirname(t2_img_path)):
            os.mkdir(os.path.dirname(t2_img_path))
        np.save(t2_img_path, t2)
        
        if not os.path.isdir(os.path.dirname(refined_seg_path)):
            os.mkdir(os.path.dirname(refined_seg_path))
        np.save(refined_seg_path, seg_pred)
        
        if not os.path.isdir(os.path.dirname(t2_projected_path)):
            os.mkdir(os.path.dirname(t2_projected_path))
        np.save(t2_projected_path, visualization)

        if not os.path.isdir(os.path.dirname(t2_region_json_path)):
            os.mkdir(os.path.dirname(t2_region_json_path))
            
        with open(t2_region_json_path, 'w') as fp:
            json.dump(avg_vals_dict, fp)
            

def run_inference(expert_pd = None, 
                  to_segment_pd = None, 
                  model_weight_file = '/data/kevin_data/checkpoints/checkpoint_weightsOnly_echo1_nomalization_quartile_dropOut0_augTrue_epoch17_trained_on_48_patients_valLoss0.8243473847938046.h5'):
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
    
        























