"""
Functions to calculate the Hausdorff chirality measure with a combination of
grid search for random initialization and BFGS search of the local minima

To Do:
-   Optimize grid search maybe, e.g., don't run BFGS at each point but 
    first do the grid search and then the local search only at the deepest
    global value 
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

# ----------------- BEGIN CODE ------------------

def scan_hausdorff_local_in_global(img):
    '''
    Optimizes for minimum Hausdorff distance by doing a grid search 
    and running BFGS at each point of this grid
    '''

    a_ = [-1.0, -0.5, 0.0, 0.5]
    params = []
    funs = []
    results = []
    inits = []

    for a in tqdm(a_):

        init = (0, 0, a)
        param = minimize(H, 
                        x0 = init, 
                        args = (img), 
                        method = 'BFGS',
                        tol = 1e-3,
                        jac = '3-point',
                        options = {'disp':False,
                                    'gtol':1e-3,
                                    'finite_diff_rel_step':0.2,
                                    'return_all':False})
        
        results.append(param)
        params.append(param.x)
        funs.append(param.fun)
        inits.append(init)

    print('Initialization screening completed')

    a_min = np.argmin(np.asarray(funs))

    print(f'Global minimum found for initialization: {denormalize_param(inits[a_min], img)}')
    print('Results for best initialization:')
    print(results[a_min])

    return results[a_min], results

# Define the function to optimize
# to_optimize: (u, v, a) in normalized between [-1 and 1]
# with u, v C [- img.shape / 2; + img.shape / 2], a C [-180, 180]

def scan_hausdorff_local(img, distance, init, step_rot = 2, step_translation = 2):
    '''
    Optimizes starting from a single initialization point
    '''
    
    print(f'minimizing the chirality distance from initialization: {init}')

    init_scaled = scale_parameters(*init, step_rot, step_translation)

    results = minimize(chirality_distance, 
                    x0 = init_scaled, 
                    args = (img, distance, step_rot, step_translation), 
                    method = 'L-BFGS-B',
                    bounds = [(0.99995, 1.00005), (0.99995, 1.00005), (0.9982,1.0018)], #Need to bound to make sure there's an overlap for IoU calculation
                    jac = None,
                    options = {
                        'disp':0,
                        'gtol':1e-5,
                        'ftol':1e-5,
                        'eps':1e-5})
    
    # To try: if does not converge, try different scaling factors automatically

    # print(f'minimization completed after {results.nit} iterations')
    print(f'Best parameters: {[round(p, 2) for p in descale_parameters(*results.x, step_rot, step_translation)]} at {round(results.fun, 2)}')
    # print(results)

    # return results

def scale_parameters(i, j, k, step_rot, step_translation):
    '''
    Method to scale parameters before optimization.
    Aims to center each parameter so that:
    - a 0 pixel translation is at 1
    - if a ~ 0: 0° ~ 1 
    - if a ~ 180: 180° ~ 1 /!\ TO-DO
    
    and so that a change of 1e-5 will:
    - translate by step pixels
    - rotate by step

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
        (k * 1e-5 / step_rot) + 1) 

def descale_parameters(i, j, k, step_rot, step_translation):
    '''
    Inverse method to scale_parameters
    '''
    return (
        (i - 1) * step_translation / 1e-5, 
        (j - 1) * step_translation / 1e-5,
        (k - 1) * step_rot / 1e-5)

def scale_range(range_param, step):
    '''
    Method to calculate a scaled range from vx and ° parameters
    In this range, one step scaled will be 1e-5 (the step size for gradient evaluation)

    Variables:
    range_param float, range of parameter, e.g., half image with
    step        float, step in ° or px

    Returns:
    range_scaled    range scaled so that one step is 1e-5
    '''
    return (range_param * 1e-5) / step

def chirality_distance(to_optimize, img, distance, step_rot, step_translation):

    # u, v, a = denormalize_param(to_optimize, img)
    u, v, a = descale_parameters(*to_optimize, step_rot, step_translation)

    # print(u,v,a)
    
    # Mirror image
    img_mirror = np.flip(img, axis = 1)

    # Transform image by translation and rotation
    # This is what needs to be optimized

    # Rotation
    img_mirror = rotate(img_mirror, angle = a)
    # Translation
    tform = EuclideanTransform(translation = (u, v))
    img_mirror = warp(img_mirror, tform, order = 0, mode = 'constant')

    if distance == 'IoU':
        return round(1.0/IoU(img, img_mirror), 6) # 1/IoU converges better than 1-IoU
    elif distance == 'Hausdorff':
        return round(hausdorff_distance_scipy(img, img_mirror), 6)
    else:
        raise ValueError('Distance must be IoU or Hausdorff')


def split_equally(length, n_idx):

    chunk_size = length // (n_idx + 1)
    return [(i + 1) * chunk_size for i in range(n_idx)]

def denormalize_param(x, img):
    # This is done to scale the parameters to optimize (translation and rotation)
    # roughly between [0,1], otherwise the search is unbalanced

    u, v, a = np.asarray(x)
    
    row = img.shape[0] / 2
    return u*row, v*row, a*180

def H_for_multiprocessing(var):

    img = var[0]
    idx = var[1][0]
    params = var[1][1]


     # Mirror image
    img_mirror = np.flip(img, axis = 1)

    # Rotation
    img_mirror = rotate(img_mirror, angle = params[2])

    # Translation
    tform = EuclideanTransform(translation = (params[0], params[1]))
    img_mirror = warp(img_mirror, tform, order = 0, mode = 'constant')
    
    return np.array([idx, 1/IoU(img, img_mirror)])


def hausdorff_distance_scipy(img1, img2):

    img1 = np.transpose(np.nonzero(img1))
    img2 = np.transpose(np.nonzero(img2))

    return max(directed_hausdorff(img1, img2)[0], directed_hausdorff(img2, img1)[0])

def IoU(img1, img2):

    img1 = np.asarray(img1, dtype = bool)
    img2 = np.asarray(img2, dtype = bool)

    intersection = img1 * img2 # logical OR
    union = img1 + img2 # logical AND

    return intersection.sum() / union.sum()

def make_contour(to_plot, k, l):
    """
    reorganize results from grid search for plotting in a contour plot

    Variables:
    to_plot         2D array [idx, u, v, alpha, distance]
    k               indice of param to plot on x axis
    l               indice of param to plot on y axis

    Returns:
    meshgrid and data
    """

    to_plot = to_plot[np.argsort(to_plot[:,0])]
    params = [np.unique(to_plot[:,i+1]) for i in range(3)]
    n_params = [len(i) for i in params]

    axes = np.meshgrid(
        params[l], # u
        params[k]  # v
    )
    contour = np.reshape(to_plot[:,-1], n_params)

    return axes, contour[:,:,0]

def plot_grid_search(to_plot, iter_param):
    """
    2D scatter with 1 x point per point of the search
    """
    # Sample data
    to_plot = to_plot[np.argsort(to_plot[:,0])]
    y = to_plot[:,-1]
    x = to_plot[:,0]
    labels = np.array([' '.join(map(str, tup)) for tup in iter_param])

    # Scaling factor for dot size and label fontsize
    scale_factor = 100

    # Create the scatter plot
    plt.figure(figsize=(15, 4))
    plt.scatter(
        x, 
        y, 
        alpha= 0.3)

    # Add labels to 5 lowest distance
    idx = np.argsort(y)[:5]

    # Set labels and title
    plt.xticks(x[idx], labels[idx], rotation = 90)
    plt.xlabel('Parameters')
    plt.yscale('log')
    plt.ylabel('log(chiraliy distance)')

    # Show plot
    plt.grid(False)
    plt.show()

def get_edges(img):
    edges = sobel(img)
    return edges > threshold_otsu(edges)



    