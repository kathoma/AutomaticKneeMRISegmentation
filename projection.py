import sys
import math
import json
import numpy as np 
import matplotlib.pyplot as plt
from scipy import optimize
from PIL import Image

def calc_R(x,y, xc, yc):
    "distance of each 2D points from the center (xc, yc)"
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def circle_distance(c, x, y):
    "algebraic distance between the data points and the mean circle centered at c=(xc, yc)"
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def leastsq_circle(x,y):
    """
    Computes leas squared circle fit based on coordinatex *x*, *y*.
    Adapted from http://wiki.scipy.org/Cookbook/Least_Squares_Circle
    """
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, _ = optimize.leastsq(circle_distance, center_estimate, args=(x, y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R, residu

def cart2pol(x, y):
    'cartesian coordinates to polar (rho, phi)'
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    'polar coordinates to cartesian (x, y)'
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def get_region_value(regiondict, binned_super, binned_deep, angle, medial_lateral_boundary_row, total_slices, slice_num):
    """
    Thresholding data from *binned_super*, *binned_deep* and updating
    *regiondict* dictionary.
    """
    if angle <=-35:
        if slice_num <= medial_lateral_boundary_row:
            regiondict['SLA'].extend(list(binned_super[:,2]))
            regiondict['DLA'].extend(list(binned_deep[:,2]))
            regiondict['LA'].extend(list(binned_super[:,2]))
            regiondict['LA'].extend(list(binned_deep[:,2]))
        else:
            regiondict['SUA'].extend(list(binned_super[:,2]))
            regiondict['DUA'].extend(list(binned_deep[:,2]))
            regiondict['UA'].extend(list(binned_super[:,2]))
            regiondict['UA'].extend(list(binned_deep[:,2]))
            
    elif -35 < angle <= 25:
        if slice_num <= total_slices//2:
            regiondict['SLC'].extend(list(binned_super[:,2]))
            regiondict['DLC'].extend(list(binned_deep[:,2]))
            regiondict['LC'].extend(list(binned_super[:,2]))
            regiondict['LC'].extend(list(binned_deep[:,2]))
        else:
            regiondict['SUC'].extend(list(binned_super[:,2]))
            regiondict['DUC'].extend(list(binned_deep[:,2]))
            regiondict['UC'].extend(list(binned_super[:,2]))
            regiondict['UC'].extend(list(binned_deep[:,2]))
            
    elif angle > 25:
        if slice_num <= total_slices//2:
            regiondict['SLP'].extend(list(binned_super[:,2]))
            regiondict['DLP'].extend(list(binned_deep[:,2]))
            regiondict['LP'].extend(list(binned_super[:,2]))
            regiondict['LP'].extend(list(binned_deep[:,2]))
        else:
            regiondict['SUP'].extend(list(binned_super[:,2]))
            regiondict['DUP'].extend(list(binned_deep[:,2]))
            regiondict['UP'].extend(list(binned_super[:,2]))
            regiondict['UP'].extend(list(binned_deep[:,2]))
            
def bootstrap_region(t2_list, reps = 1000):
    arr_of_means = np.zeros(reps)
    for i in np.arange(reps):
        arr_of_means[i]=np.mean(np.random.choice(t2_list, size=len(t2_list), replace=True))
    standard_error = np.std(arr_of_means)
    return standard_error
    
def projection(masked_cartilage, 
               thickness_div = 0.5, 
               values_threshold = 100,
               angular_bin = 5, 
               region_stat = 'mean',
               fig = False):
    '''
    This fitting femoral cartilage to a cylinder, then binning every *angular_bin*
    degrees all pixels.
    IN:
        masked_cartilage - matrix with masked cartilage T2 values
        thickness_div - a proportion of cartilage one considers 'deep' (default: 0.5)
        values_threshold - threshold to eliminate unreasonably high pixels (default: 100)
        angular_bin - how densly sample the cylinder fit for region averaging (default: 5)
        fig - logical value, whether to plot figures or not (default: False)
    OUT:
        dictionary with fields encoding averaged values of cartilage T2 map
    '''
    masked = masked_cartilage
    
    # Remove empty slices
#     masked = masked[np.sum(masked, axis = (1,2))>0]
    
    nr_slices = masked.shape[0]
    
    masked[masked > values_threshold] = 0
    mask_proj = np.nanmax(masked, axis = 0)#.max(0)

    x_coord, y_coord = np.nonzero(mask_proj)
    xc, yc, R, _ = leastsq_circle(x_coord, y_coord)

    if fig:
        plt.imshow(mask_proj)
        theta_fit = np.linspace(-np.pi, np.pi, 180)
        x_fit = xc + R * np.cos(theta_fit)
        y_fit = yc + R * np.sin(theta_fit)
        plt.plot(y_fit, x_fit, 'b-' , label="fitted circle", lw=1) #x and y swapped by Kevin
        plt.plot([yc],[xc], 'bD', mec='y', mew=1) #x and y swapped by Kevin
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(y_coord, x_coord, 'g.', label='data', mew=1) #x and y swapped by Kevin
        plt.show()

    visualization       = np.zeros((nr_slices, int(360/angular_bin)))
    super_visualization = np.zeros((nr_slices,int(360/angular_bin)))
    deep_visualization  = np.zeros((nr_slices,int(360/angular_bin)))
    thickness_map       = np.zeros((nr_slices,int(360/angular_bin)))
    min_rho_map         = np.zeros((nr_slices,int(360/angular_bin)))
    max_rho_map         = np.zeros((nr_slices,int(360/angular_bin)))
    superficial_values  = []
    deep_values         = []
    regions = ['SLA', 'SLC', 'SLP', 'SUA', 'SUC', 'SUP', 'DLA', 'DLC', 'DLP', 'DUA', 'DUC', 'DUP','LA','LC','LP','UA','UC','UP']
    regiondict = dict((i, []) for i in regions)

    mask_proj1 = masked.max(1)

    for i in range(nr_slices):
        if np.all(masked[i, :, :]==0):
            continue
        x_inds, y_inds = np.nonzero(np.squeeze(masked[i,:,:]))
        x_ind_c, y_ind_c = x_inds - xc, y_inds - yc
        values = masked[i, x_inds, y_inds] 

        rho, theta = cart2pol(x_ind_c, y_ind_c)
        theta = np.array(list(map(math.degrees, theta))) # transf. to degrees
        polar_matrix = np.vstack((theta, rho, values)).T
        
        angles = np.arange(-180, 180, angular_bin)
        
        for e, angle in enumerate(angles):
            bottom_bin, top_bin = angle, angle + angular_bin
            splice_matrix = np.logical_and(theta > bottom_bin, theta <= top_bin)
            if np.all(splice_matrix == False):
                continue
            polar_in_cone = polar_matrix[splice_matrix]
            max_radius = np.max(polar_in_cone[:,1])
            min_radius = np.min(polar_in_cone[:,1])

            cart_thickness = max_radius - min_radius
            rad_division = min_radius + cart_thickness * thickness_div

            splice_deep = polar_in_cone[:,1] <= rad_division
            binned_deep  = polar_in_cone[splice_deep]
            binned_super = polar_in_cone[~splice_deep]

            superficial_values.extend(list(binned_super[:,2]))
            deep_values.extend(list(binned_deep[:,2]))
#             get_region_value(regiondict, binned_super, binned_deep, angle, nr_slices, i)
            try:
                visualization[i, e] = np.mean(polar_in_cone[:,2])
            except:
                print("A")
            try:
                super_visualization[i, e] = np.mean(binned_super[:,2])
            except:
                print("B")
            try:
                deep_visualization[i, e] = np.mean(binned_deep[:,2])
            except:
                print("C")
            thickness_map[i,e] = cart_thickness
            min_rho_map[i,e] = min_radius
            max_rho_map[i,e] = max_radius
            
    # Iterate through all slices and angles again to calculate regional values
    vis_binary = visualization > 0
    anterior_boundary_column = int((180-35)/angular_bin)
    medial_lateral_boundary_row = np.median(np.nonzero(vis_binary[:,anterior_boundary_column]))
    
    for i in range(nr_slices):
        if np.all(masked[i, :, :]==0):
            continue
        x_inds, y_inds = np.nonzero(np.squeeze(masked[i,:,:]))
        x_ind_c, y_ind_c = x_inds - xc, y_inds - yc
        values = masked[i, x_inds, y_inds] 

        rho, theta = cart2pol(x_ind_c, y_ind_c)
        theta = np.array(list(map(math.degrees, theta))) # transf. to degrees
        polar_matrix = np.vstack((theta, rho, values)).T
        
        angles = np.arange(-180, 180, angular_bin)
        
        for e, angle in enumerate(angles):
            bottom_bin, top_bin = angle, angle + angular_bin
            splice_matrix = np.logical_and(theta > bottom_bin, theta <= top_bin)
            if np.all(splice_matrix == False):
                continue
            polar_in_cone = polar_matrix[splice_matrix]
            max_radius = np.max(polar_in_cone[:,1])
            min_radius = np.min(polar_in_cone[:,1])

            cart_thickness = max_radius - min_radius
            rad_division = min_radius + cart_thickness * thickness_div

            splice_deep = polar_in_cone[:,1] <= rad_division
            binned_deep  = polar_in_cone[splice_deep]
            binned_super = polar_in_cone[~splice_deep]

            get_region_value(regiondict, binned_super, binned_deep, angle, medial_lateral_boundary_row, nr_slices,i)
            


    averagedT2 = masked.sum()/np.count_nonzero(masked)
    avg_vals_dict = {}
    avg_vals_dict['all'] = averagedT2

    if region_stat == 'mean':
        avg_vals_dict['superficial'] = np.mean(superficial_values)
        avg_vals_dict['deep'] = np.mean(deep_values)

        for k in regiondict:
            avg_vals_dict[k] = np.mean(regiondict[k])
    
    elif region_stat == 'median':
        avg_vals_dict['superficial'] = np.median(superficial_values)
        avg_vals_dict['deep'] = np.median(deep_values)

        for k in regiondict:
            avg_vals_dict[k] = np.median(regiondict[k])
    
#     standard_error_dict = {}
#     standard_error_dict['all'] = bootstrap_region(masked[masked!=0])
#     standard_error_dict['superficial'] = bootstrap_region(superficial_values)
#     standard_error_dict['deep'] = bootstrap_region(deep_values)

#     for k in regiondict:
#         standard_error_dict[k] = bootstrap_region(regiondict[k])
    
    if fig:
        print('All : {:.2f}'.format(averagedT2))
        print('Super : {:.2f}'.format(np.mean(superficial_values)))
        print('Deep : {:.2f}'.format(np.mean(deep_values)))
        print('-'*5)
        for k in regiondict:
            print('{} : {:.2f}'.format(k, np.mean(regiondict[k])))



        
    return visualization, thickness_map, min_rho_map, max_rho_map, avg_vals_dict#,standard_error_dict
    