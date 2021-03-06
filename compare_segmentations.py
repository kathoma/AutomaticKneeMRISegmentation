import numpy as np
import pydicom
import os
import nibabel as nib
import pandas as pd
import json
import argparse
from scipy.stats import spearmanr, pearsonr, ttest_1samp
from calculate_t2 import *
from segmentation_refinement import *
from projection import *
from utils import whiten_img
from models import *
from loss_functions import dice_loss_test_volume, jaccard, coefficient_of_variation, cohen_d


def get_dice_scores(list1, list2, results_path=None):
    '''
    list1 and list2 are lists of full paths to 3D numpy arrays of binary segmentation masks of shape (slices, height, width)
    '''
    assert len(list1)==len(list2), "list1 and list2 should have the same length"
    
    f1_list = np.empty(len(list1))
    for i,path1 in enumerate(list1):
        seg1 = np.load(path1)
        seg2 = np.load(list2[i])
        d = dice_loss_test_volume(seg1,seg2)
        f1_list[i]=d
    
    if results_path:
        np.savetxt(results_path, np.array(f1_list), delimiter=",")
    else:
        return f1_list
    
def get_jaccard_indices(list1, list2, results_path=None):
    '''
    list1 and list2 are lists of full paths to 3D numpy arrays of binary segmentation masks of shape (slices, height, width)
    '''
    assert len(list1)==len(list2), "list1 and list2 should have the same length"
    
    jaccard_list = np.empty(len(list1))
    for i,path1 in enumerate(list1):
        seg1 = np.load(path1)
        seg2 = np.load(list2[i])
        j = jaccard(seg1,seg2)
        jaccard_list[i]=j
    
    if results_path:
        np.savetxt(results_path, np.array(f1_list), delimiter=",")
    else:
        return jaccard_list
    

def compare_segmentation_masks(source1_pd, source2_pd, display = True):
    source1_seg = np.array(source1_pd.refined_seg_path) 
    source2_seg = np.array(source2_pd.refined_seg_path) 

    # Get Dice Score 
    dice_scores = get_dice_scores(source1_seg, source2_seg)

    # Get Jaccard Index
    jaccard_indices = get_jaccard_indices(source1_seg, source2_seg)

    # Display results
    if display:
        print(np.round(np.mean(dice_scores),4), "+/-",np.round(np.std(dice_scores),4))
        plt.hist(dice_scores, bins = 10)
        plt.title("Dice Scores: Mean = " + str(np.round(np.mean(dice_scores),4)) + " +/- " + str(np.round(np.std(dice_scores),4)))
        plt.show()

        print(np.round(np.mean(jaccard_indices),4), "+/-",np.round(np.std(jaccard_indices),4))
        plt.hist(jaccard_indices, bins = 10)
        plt.title("Jaccard Indices: Mean = " + str(np.round(np.mean(jaccard_indices),4)) + " +/- " + str(np.round(np.std(jaccard_indices),4)))
        plt.show()
    return dice_scores, jaccard_indices


def plot_bland_altman(array1, array2, title, save = False):
    xlow = 35
    xhigh = 80
    ylow = -4
    yhigh = 4
    diff = array2 - array1
    mean = np.mean([array1, array2], axis=0)
    plt.scatter(mean,diff)
    
    # Best fit line
#     w = np.linalg.lstsq(np.hstack((mean.reshape(-1,1), np.ones((len(mean),1)))), diff)[0]
#     xx = np.linspace(*plt.gca().get_xlim()).T
#     plt.plot(xx, w[0]*xx + w[1], '-b')
    
#     plt.plot([np.min(mean),np.max(mean)], [np.mean(diff), np.mean(diff)])
#     plt.plot([np.min(mean),np.max(mean)], [np.mean(diff)+(1.96*np.std(diff)), np.mean(diff)+(1.96*np.std(diff))],'--')
#     plt.plot([np.min(mean),np.max(mean)], [np.mean(diff)-(1.96*np.std(diff)), np.mean(diff)-(1.96*np.std(diff))],'--')

#     plt.text(np.min(mean), 
#              np.mean(diff)+.1, 
#              str(np.round(np.mean(diff),2)) + ' (p='+ str(np.round(ttest_1samp(diff,0)[-1],2))+')',
#              verticalalignment = 'bottom')
    
#     plt.text(np.min(mean), 
#              np.mean(diff)+(1.96*np.std(diff))-.1, 
#              str(np.round(np.mean(diff)+(1.96*np.std(diff)),2)),
#              verticalalignment = 'top')
    
#     plt.text(np.min(mean), 
#              np.mean(diff)-(1.96*np.std(diff))+.1,
#              str(np.round(np.mean(diff)-(1.96*np.std(diff)),2)),
#              verticalalignment = 'bottom')
    
    plt.text(xlow+(.7*(xhigh-xlow)), 
             np.mean(diff) +.1, 
             str(np.round(np.mean(diff),2)) + ' (p='+ str(np.round(ttest_1samp(diff,0)[-1],2))+')',
             verticalalignment = 'bottom',
             fontsize = 13)
    
    plt.text(xlow+(.7*(xhigh-xlow)), 
             np.mean(diff)+(1.96*np.std(diff))+.1, 
             str(np.round(np.mean(diff)+(1.96*np.std(diff)),2)),
             verticalalignment = 'bottom',
             fontsize = 13)
    
    plt.text(xlow+(.7*(xhigh-xlow)), 
             np.mean(diff)-(1.96*np.std(diff))-.1,
             str(np.round(np.mean(diff)-(1.96*np.std(diff)),2)),
             verticalalignment = 'top',
             fontsize = 13)

    plt.plot([xlow,xhigh], [np.mean(diff), np.mean(diff)])
    plt.plot([xlow,xhigh], [np.mean(diff)+(1.96*np.std(diff)), np.mean(diff)+(1.96*np.std(diff))],'--')
    plt.plot([xlow,xhigh], [np.mean(diff)-(1.96*np.std(diff)), np.mean(diff)-(1.96*np.std(diff))],'--')
    


    plt.xlabel("Mean T2 Value (ms)",fontsize = 13)
    plt.ylabel("T2 Difference (ms)",fontsize = 13)
    plt.title(title,fontsize = 15)
    plt.xlim([xlow,xhigh])
    plt.ylim([ylow,yhigh])
    
    if save:
        plt.savefig('./BlandAltmanFigs/' +title + '.png', dpi=150, bbox_inches='tight')

    
    plt.show()
    

def mean_bootstrap_ci(arr, ci = 95, num_iterations = 1000):
    '''
    95% confidence interval calculated via bootstrap
    '''
    num = len(arr)
    mean_list = []
    for i in range(num_iterations):
        mu = np.mean(np.random.choice(arr, size=num, replace=True))
        mean_list.append(mu)
        
    low_percentile = (100-ci)/2
    high_percentile = 100 - low_percentile
    low = np.percentile(mean_list, low_percentile)
    high = np.percentile(mean_list, high_percentile)
    return (low, high)
     
    

def compare_region_means(list1, list2, results_path=None, correlation_method = 'pearson', bland_altman=False):
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
    ttest_dict = {}
    cv_dict = {}
    cohen_d_dict = {}
    for r in regions:
        if correlation_method == 'pearson':
            correlation_dict[r] = pearsonr(aggregate_dict[1][r],aggregate_dict[2][r])
        elif correlation_method == 'spearman':
            correlation_dict[r] = spearmanr(aggregate_dict[1][r],aggregate_dict[2][r])
        if bland_altman:
            plot_bland_altman(aggregate_dict[1][r], aggregate_dict[2][r], title="Bland Altman Plot:\n T2 value estimates in " + r + " region", save = True)

        diff = aggregate_dict[1][r] - aggregate_dict[2][r]
        ttest_dict[r] = ttest_1samp(diff,0)
        mean_abs_diff_dict[r] = (np.mean(np.abs(aggregate_dict[1][r] - aggregate_dict[2][r])),
                                 np.std(np.abs(aggregate_dict[1][r] - aggregate_dict[2][r])),
                                 mean_bootstrap_ci(np.abs(aggregate_dict[1][r] - aggregate_dict[2][r])))
        
        cv_dict[r] = coefficient_of_variation(aggregate_dict[1][r], aggregate_dict[2][r])
        cohen_d_dict[r] = cohen_d(aggregate_dict[1][r], aggregate_dict[2][r])
    
    if results_path:
        with open(os.path.join(results_path,'correlation'), 'w') as fp:
                json.dump(correlation_dict, fp)
        with open(os.path.join(results_path,'mean_abs_diff'), 'w') as fp:
                json.dump(mean_abs_diff_dict, fp)
        with open(os.path.join(results_path,'coefficient_of_variation'), 'w') as fp:
                json.dump(cv_dict, fp)
        with open(os.path.join(results_path,'cohen_d'), 'w') as fp:
                json.dump(cohen_d_dict, fp)
    else:    
        return correlation_dict, mean_abs_diff_dict, cv_dict, ttest_dict, cohen_d_dict
  
    
def compare_region_changes(list1a, list1b, list2a, list2b, results_path=None, correlation_method = 'pearson', bland_altman=False):
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
    cv_dict = {}
    cohen_d_dict = {}
    ttest_dict = {}
    for r in regions:
        if correlation_method == 'pearson':
            correlation_dict[r] = pearsonr(aggregate_dict[1][r],aggregate_dict[2][r])
        elif correlation_method == 'spearman':
            correlation_dict[r] = spearmanr(aggregate_dict[1][r],aggregate_dict[2][r])
        if bland_altman:
            plot_bland_altman(aggregate_dict[1][r], aggregate_dict[2][r], title="Bland Altman Plot:\n T2 change estimates in " + r + " region", save = False)
            
        mean_abs_diff_dict[r] = (np.mean(np.abs(aggregate_dict[1][r] - aggregate_dict[2][r])),np.std(np.abs(aggregate_dict[1][r] - aggregate_dict[2][r])))
        
        cv_dict[r] = coefficient_of_variation(aggregate_dict[1][r], aggregate_dict[2][r])
        cohen_d_dict[r] = cohen_d(aggregate_dict[1][r], aggregate_dict[2][r])
        
        diff = aggregate_dict[1][r] - aggregate_dict[2][r]
        ttest_dict[r] = ttest_1samp(diff,0)
        
    if results_path:
        with open(os.path.join(results_path,'correlation'), 'w') as fp:
            json.dump(correlation_dict, fp)
        with open(os.path.join(results_path,'mean_abs_diff'), 'w') as fp:
                json.dump(mean_abs_diff_dict, fp)
        with open(os.path.join(results_path,'coefficient_of_variation'), 'w') as fp:
                json.dump(cv_dict, fp)
        with open(os.path.join(results_path,'cohen_d'), 'w') as fp:
                json.dump(cohen_d_dict, fp)
    
    else:    
        return correlation_dict, mean_abs_diff_dict , cv_dict, ttest_dict, aggregate_dict, cohen_d_dict   
    
    
# parser = argparse.ArgumentParser(description='Compare results derived from two different segmentation methods.')

# parser.add_argument('--region_comparison_csv', 
#                     metavar='csv_file_name', 
#                     type=str,
#                     help='full paths to region_means json files. If CSV has two colums, columns are assumed to be timepoint1_segmentationMethod1 and timepoint1_segmentationMethod2. If CSV has four colums, columns are assumed to be timepoint1_segmentationMethod1, timepoint2_segmentationMethod1, timepoint1_segmentationMethod2, and timepoint2_segmentationMethod2.'
#                     default = None)

# parser.add_argument('--results_path', 
#                     metavar='results_path', 
#                     type=str,
#                     help='full path to directory where you want the results saved',
#                     default = None)

# args = parser.parse_args()

# # FINISH EVERYTHING BELOW!!
# num_columns = len(pd.read_csv(args.region_comparison_csv).columns) 
# file_array = pd.read_csv(to_segment_csv,
#                              header=0,
#                              names = ['img_dir','seg_path','refined_seg_path','t2_img_path','t2_region_json_path'])

# # Get expert segmentations for comparision purposes, if user specifies a directory of segmentations
# if args.expert_csv:
#     print("--------------------------------")
#     print("Processing expert segmentations")
#     print("--------------------------------")
#     print()
#     process_expert_segmentations(args.expert_csv)
    
# if args.to_segment_csv:
#     print("--------------------------------")
#     print("Automatically segmenting images")
#     print("--------------------------------")
#     print()
#     model_segmentation(args.to_segment_csv)
