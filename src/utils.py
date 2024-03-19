"""
Functions to calculate the Hausdorff chirality measure with a combination of
grid search for random initialization and BFGS search of the local minima
"""

import matplotlib.pyplot as plt
import imageio as iio
import numpy as np
import mrcfile
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


# ----------------- BEGIN CODE ------------------

def read_data(filename):

    filename = repr(filename)[1:-1]

    if filename.endswith('.tif') or filename.endswith('.tiff'):
        return iio.volread(filename)
    elif filename.endswith('.rec') or filename.endswith('.mrc'):
        with mrcfile.open(filename, permissive = True) as mrc:
            return np.roll(mrc.data, 1)
    else:
        raise ValueError('file format must be .rec, .mrc or .tif(f)')


def split_equally(length, n_idx):

    chunk_size = length // (n_idx + 1)
    return [(i + 1) * chunk_size for i in range(n_idx)]


def shape_error(img1, img2):

    img1 = np.asarray(img1, dtype = bool)
    img2 = np.asarray(img2, dtype = bool)

    intersection = img1 * img2 # logical AND
    union = img1 + img2 # logical OR

    return (union.sum() - intersection.sum()) / (img1.sum()*2)

def hausdorff_distance_scipy(img1, img2):

    img1 = np.transpose(np.nonzero(img1))
    img2 = np.transpose(np.nonzero(img2))

    return max(directed_hausdorff(img1, img2)[0], directed_hausdorff(img2, img1)[0])

def IoU(img1, img2):

    img1 = np.asarray(img1, dtype = bool)
    img2 = np.asarray(img2, dtype = bool)

    intersection = img1 * img2 # logical AND
    union = img1 + img2 # logical OR

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

    # Create the scatter plot
    # plt.figure(figsize=(15, 4))
    plt.scatter(
        x, 
        y, 
        alpha= 1)

    # Add labels to 5 lowest distance
    idx = np.linspace(0,60,4, endpoint=False, dtype = 'int8')
    #np.argsort(y)[0:-1:4]

    # Set labels and title
    plt.xticks(x[idx], labels[idx], rotation = 90)
    plt.xlabel('Parameters')
    plt.yscale('log')
    plt.ylabel('chiraliy distance')

    # Show plot
    plt.grid(False)
    #plt.show()

def get_edges(img):
    edges = sobel(img)
    return edges > threshold_otsu(edges)

def transform_3D(img, shifts = (0, 0, 0), angles = (0, 0, 0)):

    angles = np.deg2rad(np.array(angles))
    center = np.array(img.shape) // 2
    shifts = -center - np.array(shifts)

    tform1 = EuclideanTransform(translation = center, rotation = angles, dimensionality = 3)
    tform2 = EuclideanTransform(translation = shifts, dimensionality = 3)
    tform = tform2 + tform1
    
    return affine_transform(img, tform, order = 0, mode = 'constant')

def show_result_BFGS(img, param_mirror, V = None):
    '''
    Method to display the original and optimized mirror
    '''
    img_mirror = np.flip(img, axis = 0)
    img_mirror_t = transform_3D(img_mirror, shifts = param_mirror[:3], angles = param_mirror[3:])
    center = np.array(img.shape) // 2

    if V:
        V.add_image(img, 
            colormap = 'Blues', 
            interpolation3d = 'nearest', 
            blending = 'translucent',
            rendering = 'iso'
            )
        V.add_image(img_mirror, 
            colormap = 'RdPu', 
            interpolation3d = 'nearest', 
            blending = 'translucent',
            rendering = 'iso'
            )
        V.add_image(img_mirror_t, 
            colormap = 'Oranges', 
            interpolation3d = 'nearest', 
            blending = 'translucent',
            rendering = 'iso'
            )
        
    else:
        plt.figure(figsize = (10,4))
        plt.subplot(1,3,1)
        r = np.zeros_like(img[center[0],...], dtype = 'int8')
        plt.imshow(np.dstack((img[center[0],...], img_mirror[center[0],...], r)))
        plt.subplot(1,3,2)
        r = np.zeros_like(img[:,center[1],:], dtype = 'int8')
        plt.imshow(np.dstack((img[:,center[1],:], img_mirror[:,center[1],:], r)))
        plt.subplot(1,3,3)
        r = np.zeros_like(img[...,center[2]], dtype = 'int8')
        plt.imshow(np.dstack((img[...,center[2]], img_mirror[...,center[2]], r)))