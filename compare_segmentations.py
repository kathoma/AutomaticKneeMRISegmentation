import numpy as np
from lib import pydicom
import os
import nibabel as nib
import pandas as pd
import json
import argparse
from scipy.stats import spearmanr, pearsonr
from calculate_t2 import *
from segmentation_refinement import *
from projection import *
from utils import whiten_img
from models import *
from loss_functions import dice_loss_test_volume, jaccard

def get_dice_scores(list1, list2, results_path=None):
    '''
    list1 and list2 are lists of full paths to 3D numpy arrays of binary segmentation masks of shape (slices, height, width)
    '''
    assert len(list1)==len(list2), "list1 and list2 should have the same length"
    
    f1_list = numpy.empty(len(list1))
    for i,path1 in enumerate(list1):
        seg1 = np.load(path1)
        seg2 = np.load(list2[i])
        d = dice_loss_test_volume(seg1,seg2)
        f1_list[i]=d
    
    if results_path:
        np.savetxt(results_path, np.array(f1_list), delimiter=",")
    else:
        return f1_list


def compare_region_means(list1, list2, results_path=None):
    '''
    list1 and list2 are lists of json files whose keys are cartilage region names (str) and values are the mean T2 value in that region
    '''
    assert len(list1)==len(list2), "list1 and list2 should have the same length"
    
    with open(list1[0], "r") as read_file:
            temp_dict = json.load(read_file)
            
    regions = list(temp_dict.keys())
    num_subjects = len(list1)
    
    aggregate_dict = {}
    aggregate_dict[1]={}
    aggregate_dict[2]={}
    for r in regions:
        aggregate_dict[1][r]=np.ones(num_subjects)
        aggregate_dict[2][r]=np.ones(num_subjects)
    
    for d in range(num_subjects):
        with open(list1[d], "r") as read_file:
            dict1 = json.load(read_file)
        with open(list2[d], "r") as read_file:
            dict2 = json.load(read_file)
                    
        for r in regions:
            aggregate_dict[1][r][d] = dict1[r]
            aggregate_dict[2][r][d] = dict2[r]
     
    correlation_dict={}
    mean_abs_diff_dict = {}
    for r in regions:
        correlation_dict[r] = pearsonr(aggregate_dict[1][r],aggregate_dict[2][r])
        mean_abs_diff_dict[r] = np.mean(np.abs(aggregate_dict[1][r] - aggregate_dict[2][r]))
    
    if results_path:
        with open(os.path.join(results_path,'correlation'), 'w') as fp:
                json.dump(correlation_dict, fp)
        with open(os.path.join(results_path,'mean_abs_diff'), 'w') as fp:
                json.dump(mean_abs_diff_dict, fp)
    else:    
        return correlation_dict, mean_abs_diff_dict
  
    
def compare_region_changes(list1a, list1b, list2a, list2b, results_path=None):
    '''    
    All inputs are lists of json files whose keys are cartilage region names (str) and values are the mean T2 value in that region
    
    list1a is the baseline region measurements calculated using segmentation source 1
    list1b is the follow-up region measurements calculated using segmentation source 1
    
    list2a is the baseline region measurements calculated using segmentation source 2
    list2b is the follow-up region measurements calculated using segmentation source 2
    
    '''
    assert len(list1a)==len(list2a), "All 4 list inputs need to be the same length"
    assert len(list1a)==len(list2b), "All 4 list inputs need to be the same length"
    assert len(list1b)==len(list2a), "All 4 list inputs need to be the same length"
    assert len(list1b)==len(list2b), "All 4 list inputs need to be the same length"
    assert len(list1a)==len(list1b), "All 4 list inputs need to be the same length"
    assert len(list2a)==len(list2b), "All 4 list inputs need to be the same length"

    with open(list1a[0], "r") as read_file:
            temp_dict = json.load(read_file)
            
    regions = list(temp_dict.keys())
    num_subjects = len(list1a)
    
    aggregate_dict = {}
    aggregate_dict[1]={}
    aggregate_dict[2]={}
    for r in regions:
        aggregate_dict[1][r]=np.ones(num_subjects)
        aggregate_dict[2][r]=np.ones(num_subjects)
    
    for d in range(num_subjects):
        with open(list1a[d], "r") as read_file:
            dict1a = json.load(read_file)
        with open(list1b[d], "r") as read_file:
            dict1b = json.load(read_file)
            
        with open(list2a[d], "r") as read_file:
            dict2a = json.load(read_file)
        with open(list2b[d], "r") as read_file:
            dict2b = json.load(read_file)
                    
        for r in regions:
            aggregate_dict[1][r][d] = dict1b[r] - dict1a[r]
            aggregate_dict[2][r][d] = dict2b[r] - dict2a[r]
     
    correlation_dict={}
    mean_abs_diff_dict = {}
    for r in regions:
        correlation_dict[r] = pearsonr(aggregate_dict[1][r],aggregate_dict[2][r])
        mean_abs_diff_dict[r] = np.mean(np.abs(aggregate_dict[1][r] - aggregate_dict[2][r]))
        
    if results_path:
        with open(os.path.join(results_path,'correlation'), 'w') as fp:
            json.dump(correlation_dict, fp)
        with open(os.path.join(results_path,'mean_abs_diff'), 'w') as fp:
                json.dump(mean_abs_diff_dict, fp)
    
    else:    
        return correlation_dict, mean_abs_diff_dict        
    
    
parser = argparse.ArgumentParser(description='Compare results derived from two different segmentation methods.')

parser.add_argument('--region_comparison_csv', 
                    metavar='csv_file_name', 
                    type=str,
                    help='full paths to region_means json files. If CSV has two colums, columns are assumed to be timepoint1_segmentationMethod1 and timepoint1_segmentationMethod2. If CSV has four colums, columns are assumed to be timepoint1_segmentationMethod1, timepoint2_segmentationMethod1, timepoint1_segmentationMethod2, and timepoint2_segmentationMethod2.'
                    default = None)

parser.add_argument('--results_path', 
                    metavar='results_path', 
                    type=str,
                    help='full path to directory where you want the results saved',
                    default = None)

args = parser.parse_args()

# FINISH EVERYTHING BELOW!!
num_columns = len(pd.read_csv(args.region_comparison_csv).columns) 
file_array = pd.read_csv(to_segment_csv,
                             header=0,
                             names = ['img_dir','seg_path','refined_seg_path','t2_img_path','t2_region_json_path'])

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
