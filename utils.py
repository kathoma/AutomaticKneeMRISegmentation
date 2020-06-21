import numpy as np
import random
import pydicom
import scipy.io as sio
import matplotlib as plt
from skimage import io, transform
import scipy.ndimage as ndimage
import math
import os

def make_dictionary(pid_set, root):    
    #################################################
    # Create a dictionary of data for each data set
    #################################################
    
    set_dict = {}
    year_dict = {}
    slice_dict = {}
    path_dict = {}
    
    for pid_counter, pid in enumerate(pid_set):
        for yr in [4,8]:
            seg_list = os.listdir(os.path.join(root , "BinarySegmentations_perSlice_rad", pid))
            seg_list = np.array(seg_list)
            seg_sliceNum = [int(k.split('_')[2].split('.')[0]) for k in seg_list]
            index = np.argsort(seg_sliceNum)
            seg_list = seg_list[index]
            
            seg_year = [k.split('_')[1] for k in seg_list]
            seg_year = np.array(seg_year)
            year_check = seg_year == str(yr)

            seg_list = seg_list[year_check]
            num_slices = len(seg_list)
            
            mri_list = os.listdir(os.path.join(root ,  "YR" + str(yr) , pid , "T2"))
            mri_list = np.array(sorted(mri_list, key=lambda x: int(x)))
            assert(num_slices == len(mri_list)/7)
                
#             mri_list = np.sort(mri_list)
#             mri_list = mri_list.astype(int)
#             mri_list = np.sort(mri_list)
#             print(mri_list)
            
            for slice_num in np.arange(num_slices):
                seg_slice_path = seg_list[slice_num]
                mri_slice_paths = mri_list[slice_num::num_slices]    

                path_dict = {k: os.path.join(root , "YR" + str(yr), pid , "T2" , mri_slice_paths[k]) for k in range(7)}
                path_dict['seg'] =  os.path.join(root ,"BinarySegmentations_perSlice_rad" , pid , seg_slice_path)
                
                path_dict['cart_count'] = np.sum(sio.loadmat(path_dict['seg'])['femoral_cartilage'])
                
                slice_dict[slice_num] = path_dict
                
            year_dict[yr] = slice_dict
            slice_dict = {}
        set_dict[pid] = year_dict
        year_dict = {}
        
    return set_dict



def make_giant_mat(data_dict, normalization, echo_nums = [1], echo_aug = False): # echo_nums = [0,1,2,3,4,5,6]
    n_channels = 1
    dim = (384,384)

    seg_filenames = []     
    mri_slice_filenames = []


    # Get filenames from dictionary and root
    count = 0
    for pid, year_dict in data_dict.items():
        for year, slice_dict in data_dict[pid].items():
            for slice_num, path_dict in data_dict[pid][year].items():
                temp_mri_slice_filenames = []
                for file_key, path in data_dict[pid][year][slice_num].items():
                    if ".mat" in str(path):
                        seg_path = (path)
                    
                    elif np.isin([file_key,file_key],[0,1,2,3,4,5,6])[0]:
                        temp_mri_slice_filenames.append(path)

                # Put this slice's MRI echo images in chronological order
                temp_mri_slice_filenames.sort()
                
                # Only keep the echo relaxation times that are specified by the user
                if echo_aug:
                    # Grab an alternative echo 40% of the time and grab the normal echo 60% of the time
                    rand_num = np.random.uniform()
                    if rand_num < .2:
                        # Grab the echo that is one unit less than the current minimum if min is not 0
                        if np.min(echo_nums)>0:
                            temp_mri_slice_filenames = [temp_mri_slice_filenames[i] for i in [np.min(echo_nums)-1]]
                        else:
                            temp_mri_slice_filenames = [temp_mri_slice_filenames[i] for i in [np.min(echo_nums)+1]]
                            
                    elif rand_num > .8:
                        # Grab the echo that is one unit more than the current maximum if max is not 6
                        if np.max(echo_nums)<6:
                            temp_mri_slice_filenames = [temp_mri_slice_filenames[i] for i in [np.max(echo_nums)+1]]
                        else:
                            temp_mri_slice_filenames = [temp_mri_slice_filenames[i] for i in [np.max(echo_nums)-1]]
                            
                    else:
                        temp_mri_slice_filenames = [temp_mri_slice_filenames[i] for i in echo_nums] 
                        
                else:
                    temp_mri_slice_filenames = [temp_mri_slice_filenames[i] for i in echo_nums]
                
                

                # Add filepaths to the segmentation and mri lists
                for relax_time in np.arange(len(temp_mri_slice_filenames)):
                    seg_filenames.append(seg_path) # repeatedly add this segmentation for each timepoint
                    mri_slice_filenames.append(temp_mri_slice_filenames[relax_time])
                    count = count + 1

    num_images = len(mri_slice_filenames)
    X = np.empty((num_images, *dim, 1))
    y = np.empty((num_images, *dim, 1))

    for i in np.arange(num_images):
        mri_slice_filename = mri_slice_filenames[i]
        seg_slice_filename = seg_filenames[i]

        # Store whitened image
        img_array = np.copy(pydicom.read_file(mri_slice_filename).pixel_array)
        img_array[img_array>5000] = 0 # Get rid of outlier background pixels seen in top left corner of some images @ pixel [0,10]

        img_array_reshape = img_array.reshape((img_array.shape[0],img_array.shape[1], n_channels)) 

        if normalization == 'mean_min_max':
            img_array_centered = img_array_reshape - np.mean(img_array_reshape)
            img_array_normalized = img_array_centered / np.std(img_array_centered)
            img_clean = ((img_array_normalized - np.min(img_array_normalized)) / (np.max(img_array_normalized) - np.min(img_array_normalized))) * (2) + -1 # *(new_range) + new_min
            
        elif normalization == 'quartile':
            img_array_centered = img_array_reshape - np.median(img_array_reshape)
            img_array_normalized = img_array_centered / np.percentile(img_array_centered, 75)
            img_clean = ((img_array_normalized - np.percentile(img_array_normalized,25)) / (np.percentile(img_array_normalized,75) - np.percentile(img_array_normalized,25))) * (2) + -1
            
            img_clean[img_clean < np.percentile(img_clean,3)] = np.percentile(img_clean,3)
            img_clean[img_clean > np.percentile(img_clean,97)] = np.percentile(img_clean,97)
            

        X[i,] = img_clean 

        mask = sio.loadmat(seg_slice_filename)['femoral_cartilage']

        #             y[i,] = ndimage.binary_dilation(mask.reshape((mask.shape[0], mask.shape[1], self.n_channels)), iterations = 5)
        y[i,] = mask.reshape((mask.shape[0], mask.shape[1], n_channels))

    return X, y


def make_echo_dict(data_dict, normalization, echo_nums = [0,1,2,3,4,5,6]): # echo_nums = [0,1,2,3,4,5,6]

    echo_dict = {}
    
    # Get filenames from dictionary and root
    for pid, year_dict in data_dict.items():
        print(pid)
        
        for year, slice_dict in data_dict[pid].items():
            subject_year_dict = {}
            echo_time_dict = {}
            
            seg_filenames = []
            echo_array_list = []
            echo_array_white_list = []
            slice_count = 0
            
            for slice_num, path_dict in data_dict[pid][year].items():
                temp_mri_slice_filenames = []
                
                for file_key, path in data_dict[pid][year][slice_num].items():
                    if ".mat" in str(path):
                        seg_path = (path)
                    
                    elif np.isin([file_key,file_key],[0,1,2,3,4,5,6])[0]:
                        temp_mri_slice_filenames.append(path)

                # Put this slice's MRI echo images in chronological order
                temp_mri_slice_filenames.sort()
                
                # Only keep the echo relaxation times that are specified by the user
                temp_mri_slice_filenames = [temp_mri_slice_filenames[i] for i in echo_nums]
                
                # Make a list of echo numpy arrays for this slice
                echo_array = [pydicom.read_file(echo).pixel_array for echo in temp_mri_slice_filenames]
                echo_array_white = [whiten_img(echo, normalization) for echo in echo_array]
                # Convert the list into a larger array
                echo_array = np.stack(echo_array, axis = 0) #num_echos x 384 x 384
                echo_array_list.append(echo_array)
                echo_array_white = np.stack(echo_array_white, axis = 0) #num_echos x 384 x 384
                echo_array_white_list.append(echo_array_white)

                # Save the segmentation mask for this slice
                seg_filenames.append(seg_path) 
                
                # Save echo times for all echoes in this slice
                echo_time_dict[slice_num] = np.array([float(pydicom.read_file(echo).AcquisitionTime) for echo in temp_mri_slice_filenames])

            # For each volume, generate a 3D segmentation mask and 4D image volume (nr_slices, time_steps, width, height)
            segmentation_3Darray = [sio.loadmat(seg_slice_filename)['femoral_cartilage'] for seg_slice_filename in seg_filenames]
            segmentation_3Darray = np.stack(segmentation_3Darray, axis = 0) #num_slices x 384 x 384
            image_4Darray = np.stack(echo_array_list, axis = 0)
            image_4Darray_white = np.stack(echo_array_white_list, axis = 0)
            
            subject_year_dict['seg'] = segmentation_3Darray
            subject_year_dict['img'] = image_4Darray
            subject_year_dict['img_white'] = image_4Darray_white
            subject_year_dict['echo_times'] = echo_time_dict

            
            echo_dict[(pid,year)] = subject_year_dict


    return echo_dict

def whiten_img(img, normalization):
        img = np.copy(img)
        img_array_reshape = img.reshape((img.shape[0],img.shape[1], -1)) 
        img[img>5000] = 0 # Get rid of outlier background pixels seen in top left corner of some images @ pixel [0,10]
        
        if normalization == 'mean_min_max':
            img_array_centered = img_array_reshape - np.mean(img_array_reshape)
            img_array_normalized = img_array_centered / np.std(img_array_centered)
            img_clean = ((img_array_normalized - np.min(img_array_normalized)) / (np.max(img_array_normalized) - np.min(img_array_normalized))) * (2) + -1 # *(new_range) + new_min
            return img_clean
        
        elif normalization == 'quartile':
            img_array_centered = img_array_reshape - np.median(img_array_reshape)
            img_array_normalized = img_array_centered / np.percentile(img_array_centered, 75)
            img_clean = ((img_array_normalized - np.percentile(img_array_normalized,25)) / (np.percentile(img_array_normalized,75) - np.percentile(img_array_normalized,25))) * (2) + -1
            
            img_clean[img_clean < np.percentile(img_clean,3)] = np.percentile(img_clean,3)
            img_clean[img_clean > np.percentile(img_clean,97)] = np.percentile(img_clean,97)
            return img_clean
        
