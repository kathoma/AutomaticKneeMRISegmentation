# AutomaticKneeMRISegmentation

This repository provides code for the following manuscript (currently under review):
Open source software for automatic subregional assessment of knee cartilage degradation using quantitative T2 relaxometry and deep learning

This software provides the following automated functionality for multi-echo spin echo T2-weighted knee MRIs:
- Segmentation of femoral cartilage
- Projection of the femoral cartilage onto a 2D plane
- Division of the projected cartilage into 12 subregions along medial-lateral, superficial-deep, and anterior-central-posterior boundaries
- Calculation of the average T2 value in each subregion
- Calculation of the change in average T2 value over time for each subregion (if 2 imaging time points are available for a given person)
- Comparison of results across different readers/models

FullPipeline.ipynb walks through an example of how to use the full pipeline to analyze individual images, calculate changes in a patient over time, and compare results for segmentations from different readers. 
