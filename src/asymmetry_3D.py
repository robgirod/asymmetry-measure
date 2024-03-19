"""
Functions to calculate the Hausdorff chirality measure with a combination of
grid search for random initialization and BFGS search of the local minima

TO-DO:
- Method to get bounds for BFGS
- Method to put the grid search here
"""

import matplotlib.pyplot as plt
import imageio as iio
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import directed_hausdorff
from sklearn import decomposition
from skimage import registration
from skimage.transform import warp, downscale_local_mean, EuclideanTransform, rotate
from skimage import img_as_float32
from skimage.exposure import rescale_intensity
import mrcfile
from skimage.metrics import hausdorff_distance, hausdorff_pair
from skimage.filters import threshold_otsu, sobel
from skimage.draw import line_nd, ellipse, polygon
from tqdm import tqdm
from itertools import product
import math
from scipy.ndimage import affine_transform
from src.utils import *
from time import time
import concurrent.futures


# ----------------- BEGIN CODE ------------------

def eval_time_grid_search(Q, distance, iter_param):
    '''
    Function to evaluate the time required for a grid search
    '''
    to_CPU = [(Q, param, distance) for param in enumerate(iter_param)] # here param is [idx, (u, v, w, a, b, c)]

    start = time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(to_multiprocessing_3D, var) for var in to_CPU[:100]]
        H_global = [future.result() for future in concurrent.futures.as_completed(futures)]
    t = time()-start

    l = len(to_CPU)

    return round((t * l) / 100)

def to_multiprocessing_3D(var):
    '''
    Function to arrange variables for the multihreaded calculation of distances
    '''

    img = var[0]
    idx = var[1][0]
    u, v, w, a, b, c = var[1][1]
    distance = var[2]
 
    return np.array([idx, chirality_distance_3D((u,v,w,a,b,c), img, distance)])

def minimize_distance_local_3D(img, distance, init, step_rot = 2, step_translation = 2):
    '''
    Optimizes starting from a single initialization point
    '''
    
    print(f'minimizing the chirality distance from initialization: {init}')

    init_scaled = scale_parameters(*init, step_rot, step_translation)

    results = minimize(chirality_distance_3D, 
                    x0 = init_scaled, 
                    args = (img, distance, step_rot, step_translation), 
                    method = 'L-BFGS-B',
                    bounds = [(0.99995, 1.00005), 
                              (0.99995, 1.00005),
                              (0.99995, 1.00005), 
                              (0.9982,1.0018),
                              (0.9982,1.0018),
                              (0.9982,1.0018)], #Need to bound to make sure there's an overlap for IoU calculation
                    jac = None,
                    options = {
                        'disp':0,
                        'gtol':1e-4,
                        'ftol':1e-4,
                        'eps':1e-5})
    
    # To try: if does not converge, try different scaling factors automatically

    print(f'minimization completed after {results.nit} iterations')
    print(f'Best parameters: {[round(p, 2) for p in descale_parameters(*results.x, step_rot, step_translation)]} at {round(results.fun, 2)}')
    print(results)

    return results

def test_step_size(img, distance, init, step_rot = 2, step_translation = 2):
    '''
    Method to test the variation in a chirality distance for given step sizes
    u, v, w are tested for 1 step_translation
    a, b, c are tested for 1 step_rotation

    Best step size gives a change of ~1e-3 for SE, ~1.5-2e-2 for IoU
    It is best to have change_translate ~ change_rotation for convergence
    Note: cannot have step size < 1 px
    '''

    init_scaled = scale_parameters(*init, step_rot, step_translation)
    fun_at_init = chirality_distance_3D(init_scaled, img, distance, step_rot, step_translation)

    # Test step translation
    change_at_step_translation = 0.0
    for i in range(3): # Test step translation
        params = np.array(init)
        params[i] += step_translation
        step_param_scaled = scale_parameters(*params, step_rot, step_translation)
        fun_at_step_translation = chirality_distance_3D(
            step_param_scaled,
            img,
            distance,
            step_rot,
            step_translation
        )
        change_at_step_translation += abs(fun_at_step_translation - fun_at_init)

    print(f'Average change for a translation step of {step_translation} px:','{:e}'.format(change_at_step_translation/3))

    # Test step rotation
    change_at_step_rotation = 0.0
    for i in range(3): 
        params = np.array(init)
        params[3+i] += step_rot
        step_param_scaled = scale_parameters(*params, step_rot, step_translation)
        fun_at_step_rotation = chirality_distance_3D(
            step_param_scaled,
            img,
            distance,
            step_rot,
            step_translation
        )
        change_at_step_rotation += abs(fun_at_step_rotation - fun_at_init)


    print(f'Average change for a rotation step of {step_rot}°:','{:e}'.format(change_at_step_rotation/3))

def scale_parameters(i, j, k, l, m, n, step_rot, step_translation):
    '''
    Method to scale parameters before optimization.
    Aims to center each parameter so that:
    - a 0 pixel translation is at 1
    - if a ~ 0: 0° ~ 1 
    - if a ~ 180: 180° ~ 1 /!\ TO-DO
    
    and so that a change of 1e-5 will:
    - translate by step pixels
    - rotate by step °

    Variables:
    p               parameter to scale
    range_scaled    range scaled
    range_real      intial range

    Returns:
    p_scaled        parameter scaled
    '''

    return (
        (i * 1e-5 / step_translation) + 1, 
        (j * 1e-5 / step_translation) + 1,
        (k * 1e-5 / step_translation) + 1,
        (l * 1e-5 / step_rot) + 1,
        (m * 1e-5 / step_rot) + 1,
        (n * 1e-5 / step_rot) + 1) 

def descale_parameters(i, j, k, l, m, n, step_rot, step_translation):
    '''
    Inverse method to scale_parameters
    '''
    return (
        (i - 1) * step_translation / 1e-5, 
        (j - 1) * step_translation / 1e-5,
        (k - 1) * step_translation / 1e-5,
        (l - 1) * step_rot / 1e-5,
        (m - 1) * step_rot / 1e-5,
        (n - 1) * step_rot / 1e-5)

def chirality_distance_3D(to_optimize, img, distance, step_rot = 0, step_translation = 0):
    '''
    Method to return the chosen chirality distance after a transformation of the mirror
    image dictated by the parameters passed in to_optimized

    If called from minimize_hausdorff_local_3D, to_optimize is scaled to 1e-5 steps
    If called from to_multiprocessing_3D, to_optimized is in px and °
    '''


    if step_rot or step_translation:
        u, v, w, a, b, c = descale_parameters(*to_optimize, step_rot, step_translation)
    else:
        u, v, w, a, b, c = to_optimize
    
    # Mirror image
    img_mirror = np.flip(img, axis = 1)

    # Transform image by translation and rotation
    # This is what needs to be optimized

    img_mirror = transform_3D(img_mirror, (u, v, w), (a, b, c))

    if distance == 'IoU':
        return 1.0/IoU(img, img_mirror) # 1/IoU converges better than 1-IoU
    elif distance == 'Hausdorff':
        return round(hausdorff_distance_scipy(img, img_mirror), 6)
    elif distance == 'SE':
        return shape_error(img, img_mirror)
    else:
        raise ValueError('Distance must be IoU, SE or Hausdorff')


