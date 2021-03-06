import os
import pandas as pd
import argparse
import numpy as np
import pickle

def make_expert_csv_specific_years(pID, 
                                   years, 
                                   img_dir, 
                                   dir_to_save, 
                                   seg_provided, 
                                   seg_format = "numpy", 
                                   csv_filename='file_paths.csv'):
    
    # specific pID-year combinations
    assert len(pID) == len(years), "pID and years should have the same length"
    column_names = ['img_dir','seg_path','refined_seg_path','t2_img_path','t2_projected_path','t2_region_json_path']
    df = pd.DataFrame(columns = column_names)
    
    for i in range(len(pID)):
        img_dir_path = os.path.join(img_dir, "YR"+str(years[i]), pID[i], "T2")
        if seg_provided:
            if seg_format == "nifti":
                seg_path = os.path.join(dir_to_save, "segmentations", pID[i]+"_"+str(years[i])+"_fc.nii.gz")
            elif seg_format == "numpy":
                seg_path = os.path.join(dir_to_save, "segmentations", pID[i]+"_"+str(years[i])+".npy") 
        else:
            seg_path = os.path.join(dir_to_save, "segmentations", pID[i]+"_"+str(years[i])+".npy")
        
        segmentations_refined_path = os.path.join(dir_to_save, "segmentations_refined", pID[i]+"_"+str(years[i])+".npy")
        t2maps_path = os.path.join(dir_to_save, "t2maps", pID[i]+"_"+str(years[i])+".npy")
        t2_projected_path = os.path.join(dir_to_save, "t2_projected", pID[i]+"_"+str(years[i])+".npy")
        region_means_path = os.path.join(dir_to_save, "region_means", pID[i]+"_"+str(years[i])+".json")
        
        d = {'img_dir': [img_dir_path], 
             'seg_path': [seg_path],
             'refined_seg_path': [segmentations_refined_path],
             't2_img_path': [t2maps_path],
             't2_projected_path': [t2_projected_path],
             't2_region_json_path': [region_means_path]}
        
        row_df = pd.DataFrame(data=d)
                                    
        df = df.append(row_df)
    
    df.to_csv(path_or_buf = os.path.join(dir_to_save, csv_filename),index=False)
    return df

    
def make_expert_csv_all_years(pID, img_dir, dir_to_save, csv_filename = "file_paths.csv"):
    # both years for each pID
    column_names = ['img_dir','seg_path','refined_seg_path','t2_img_path','t2_projected_path','t2_region_json_path']
    df = pd.DataFrame(columns = column_names)
    
    for i in range(len(pID)):
        for yr in ['4','8']:
            img_dir_path = os.path.join(img_dir, "YR"+ yr, pID[i], "T2")
            seg_path = os.path.join(dir_to_save, "segmentations", pID[i]+"_"+ yr +".npy")
            segmentations_refined_path = os.path.join(dir_to_save, "segmentations_refined", pID[i]+"_"+ yr +".npy")
            t2maps_path = os.path.join(dir_to_save, "t2maps", pID[i] + "_" + yr +".npy")
            t2_projected_path = os.path.join(dir_to_save, "t2_projected", pID[i] + "_" + yr +".npy")
            region_means_path = os.path.join(dir_to_save, "region_means", pID[i] + "_" + yr +".json")

            d = {'img_dir': [img_dir_path], 
                 'seg_path': [seg_path],
                 'refined_seg_path': [segmentations_refined_path],
                 't2_img_path': [t2maps_path],
                 't2_projected_path': [t2_projected_path],
                 't2_region_json_path': [region_means_path]}
            
            row_df = pd.DataFrame(data=d)

            df = df.append(row_df)
    
    
    df.to_csv(path_or_buf = os.path.join(dir_to_save, csv_filename),index=False)
    return df
    
    
# def make_comparison_csv_one_timepoint(pID, years, img_dir, dir_to_save):

    
    
    
    
    
    
# def make_comparison_csv_two_timepoint_delta(pID, img_dir, dir_to_save):
    
    
    
    
    
    
    
