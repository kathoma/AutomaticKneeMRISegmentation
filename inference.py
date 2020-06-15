# from __future__ import print_function, division

# import sys
# sys.path.insert(0, 'lib')

import numpy as np
# import random
# # import keras as k


# import scipy.io as sio


# import pandas as pd
# from skimage import io, transform
# import matplotlib.pyplot as plt

# import pickle
# import scipy.ndimage as ndimage

# import math
# import os

# from loss_functions import dice_loss, dice_loss_test, dice_loss_test_volume
# from models import unet_2d_model
# from utils import make_giant_mat, make_dictionary, make_echo_dict

# import scipy.linalg as la
# from joblib import Parallel, delayed


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import f1_score

# from scipy.optimize import curve_fit

# from skimage import measure
# import scipy.stats as ss

# import skimage
from lib import pydicom
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
    model.load_weights(args.model_weight_file)
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

def process_expert_segmentations(expert_csv):
    expert_file_array = pd.read_csv(expert_csv,
                                    header=0,
                                    names = ['img_dir','seg_path','refined_seg_path','t2_img_path','t2_region_json_path'])
    
    
    for i in range(len(expert_file_array)):
        
        img_dir = expert_file_array.img_dir[i]
        seg_path = expert_file_array.seg_path[i]
        print(seg_path)
        
        refined_seg_path = expert_file_array.refined_seg_path[i]
        t2_img_path = expert_file_array.t2_img_path[i]
        t2_region_json_path = expert_file_array.t2_region_json_path[i]
        
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

        if not os.path.isdir(os.path.dirname(t2_region_json_path)):
            os.mkdir(os.path.dirname(t2_region_json_path))
            
        with open(t2_region_json_path, 'w') as fp:
            json.dump(avg_vals_dict, fp)
            
                      

def model_segmentation(to_segment_csv, normalization = 'quartile'):
    file_array = pd.read_csv(to_segment_csv,
                             header=0,
                             names = ['img_dir','seg_path','refined_seg_path','t2_img_path','t2_region_json_path'])
    
    for i in range(len(file_array)):
        
        img_dir = file_array.img_dir[i]
        print(img_dir)
        seg_path = file_array.seg_path[i]
        
        refined_seg_path = file_array.refined_seg_path[i]
        t2_img_path = file_array.t2_img_path[i]
        t2_region_json_path = file_array.t2_region_json_path[i]
        
        # Retrieve the corresponding MRI and echo times (slices, echoes, height, width)
        mese, t2times = assemble_4d_mese(img_dir) 
        
        # Whiten echo 1 of each slice
        mese_white = [whiten_img(s, normalization = normalization) for s in mese[:,1,:,:]]
        mese_white = np.stack(mese_white)
        
        # Get model
        model = get_model(args.model_weight_file)
        
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

        if not os.path.isdir(os.path.dirname(t2_region_json_path)):
            os.mkdir(os.path.dirname(t2_region_json_path))
            
        with open(t2_region_json_path, 'w') as fp:
            json.dump(avg_vals_dict, fp)
            
        
        
        
        
    
            
            


parser = argparse.ArgumentParser(description='Segment femoral cartilage in multi echo spin echo MRIs.')

parser.add_argument('--model_weight_file', 
                    metavar='my_file_path.h5', 
                    type=str,
                    help='an h5 model weights checkpoint',
                    default = '/data/kevin_data/checkpoints/checkpoint_weightsOnly_echo1_nomalization_quartile_dropOut0_augTrue_epoch17_trained_on_48_patients_valLoss0.8243473847938046.h5')

parser.add_argument('--expert_csv', 
                    metavar='csv_file_name', 
                    type=str,
                    help='paths to gold standard segmentations in 3D nifti format and their corresponding images',
                    default = None)

parser.add_argument('--to_segment_csv', 
                    metavar='csv_file_name', 
                    type=str,
                    help='paths to directories that each contain the slices of multi echo spin echo images',
                    default = None)

args = parser.parse_args()

# Get expert segmentations for comparision purposes, if user specifies a directory of segmentations
if args.expert_csv:
    print("--------------------------------")
    print("Processing expert segmentations")
    print("--------------------------------")
    print()
    process_expert_segmentations(args.expert_csv)
    
if args.to_segment_csv:
    print("--------------------------------")
    print("Automatically segmenting images")
    print("--------------------------------")
    print()
    model_segmentation(args.to_segment_csv)























