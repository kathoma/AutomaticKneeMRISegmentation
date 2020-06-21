import pandas as pd
import argparse
from inference_utils import process_expert_segmentations, model_segmentation
    
        

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
    expert_file_array = pd.read_csv(expert_csv,
                                    header=0,
                                    names = ['img_dir','seg_path','refined_seg_path','t2_img_path','t2_region_json_path'])

    process_expert_segmentations(expert_file_array)
    
if args.to_segment_csv:
    print("--------------------------------")
    print("Automatically segmenting images")
    print("--------------------------------")
    print()
    
    file_array = pd.read_csv(args.to_segment_csv,
                             header=0,
                             names = ['img_dir','seg_path','refined_seg_path','t2_img_path','t2_region_json_path'])
    
    model_segmentation(file_array, model_weight_file = args.model_weight_file)























