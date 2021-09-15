# Automatic subregional assessment of knee cartilage degradation

This repository provides code for the following manuscript:
"Open source software for automatic subregional assessment of knee cartilage degradation using quantitative T2 relaxometry and deep learning"
Link to paper: https://journals.sagepub.com/doi/abs/10.1177/19476035211042406?journalCode=cara

This software provides the following automated functionality for multi-echo spin echo T2-weighted knee MRIs:
- Segmentation of femoral cartilage
- Projection of the femoral cartilage onto a 2D plane
- Division of the projected cartilage into 12 subregions along medial-lateral, superficial-deep, and anterior-central-posterior boundaries
- Calculation of the average T2 value in each subregion
- Calculation of the change in average T2 value over time for each subregion (if 2 imaging time points are available for a given person)
- Comparison of results across different readers/models

FullPipeline.ipynb walks through an example of how to use the full pipeline to analyze individual images, calculate changes in a patient over time, and compare results for segmentations from different readers. 

Requires CUDA Version 9.0.176
Tested with CUDA 9.0 and cudnn 7.3.0 in ubuntu 18.04 

# Instructions for getting started
Follow these instructions for installing the appropriate version of CUDA and cudnn: https://github.com/akirademoss/cuda-9.0-installation-on-ubuntu-18.04

# Download software for creating a virtual environment, then create a new virtual environment called kneeseg and activate it:
```
pip install virtualenv
virtualenv -p /usr/bin/python3 kneeseg
source kneeseg/bin/activate
```
Download this repository onto your computer:
```
git clone https://github.com/kathoma/AutomaticKneeMRISegmentation.git
```
Enter into the directory you just downloaded:
```
cd AutomaticKneeMRISegmentation
```
Install the necessary dependencies in your new virtual environment:
```
pip install -r requirements_python3.txt
```
Make a directory for the model weights:
```
mkdir model_weights
cd model_weights
```
Download the weights for the trained model:
```
wget https://storage.googleapis.com/automatic_knee_mri_segmentation/model_weights_quartileNormalization_echoAug.h5
cd ..
```
Follow the steps in FullPipeline.ipynb to use the model. 

