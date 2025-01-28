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
from skimage.transform import warp, downscale_local_mean, EuclideanTransform, rotate, rescale
from skimage import img_as_float32
from skimage.exposure import rescale_intensity
import mrcfile
from skimage.metrics import hausdorff_distance, hausdorff_pair
from skimage.filters import threshold_otsu, sobel, median
from skimage.draw import line_nd, ellipse, polygon
from tqdm import tqdm
from itertools import product
import math
from scipy.ndimage import affine_transform


# ----------------- BEGIN CODE ------------------
def plot_histogram(vol):
    
    vol = crop_percentile_intensity(vol)

    x, y, z = np.shape(vol)
    n = np.floor(x/2).astype('int')
    
    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 2.5))

    # Plotting the original image.
    ax[0].imshow(vol[n], cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    # Plotting the histogram and the two thresholds obtained
    h, b, p = ax[1].hist(vol[vol>0].ravel(), bins=255)
    ax[1].set_title('Histogram')
    ax[1].set_yticks([])
    ax[1].set_yscale('log')
    #ax[1].set_ylim(0, max_y)

    plt.show()

def crop_percentile_intensity(vol):

    vol = vol-np.min(vol[vol > 0])
    vol = vol / np.max(vol)

    p1 = np.percentile(vol, 2)

    return np.where(vol < p1, 0, vol)

def prepare_volume(vol, thresh, footprint = None, binning = None, V = None):

    if binning: vol = downscale_local_mean(vol, binning)

    vol = crop_percentile_intensity(vol)

    x, y, z = np.shape(vol)
    n = np.floor(x/2).astype('int')

    # Threshold
    if thresh == 'otsu':
        thresh = threshold_otsu(vol)
    
    binarized = vol > thresh
    binarized = binarized.astype(int) * 255

    # Median
    if footprint:
        dim = 2 * footprint + 1
        binarized = median(binarized, footprint = np.ones((dim, dim, dim)))

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 3.5))

    # Plotting the original image.
    ax[0].imshow(vol[n], cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    # Plotting the histogram and the two thresholds obtained
    h, b, p = ax[1].hist(vol[vol>0].ravel(), bins=255)
    max_y = np.max(h[b[:-1] > thresh]) * 2
    ax[1].set_title('Histogram')
    ax[1].axvline(thresh, color='r')
    ax[1].set_yticks([])
    ax[1].set_ylim(0, max_y)

    # Plotting the threshold result.
    ax[2].imshow(binarized[n], cmap = 'gray')
    ax[2].set_title('Binarized')
    ax[2].axis('off')

    plt.subplots_adjust()
    plt.show()

    if V:
        V.add_image(binarized,
            colormap = 'gray', 
            interpolation3d = 'linear', 
            blending = 'translucent',
            rendering = 'iso',
            iso_threshold = 0
            )

    return binarized

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


def disjunctive_union(img1, img2):

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

def save_result(img, param_mirror, filename):
    '''
    Method to save the mirror and translated volume giving the lowest asymmetry
    '''
    img_mirror = np.flip(img, axis = 0)
    img_mirror_t = transform_3D(img_mirror, shifts = param_mirror[:3], angles = param_mirror[3:])

    img = (img - np.min(img)) / np.max(img)
    img_mirror_t = (img_mirror_t - np.min(img_mirror_t)) / np.max(img_mirror_t)

    achiral = np.where(img + img_mirror_t == 2, 1, 0)
    chiral = np.where(img + achiral == 1, 1, 0)

    filename = repr(filename)[1:-1]

    dot_index = filename.rfind('.')
    savename = filename[:dot_index]

    iio.volwrite(savename + '_achiral.tif', achiral.astype('uint8'))
    iio.volwrite(savename + '_chiral.tif', chiral.astype('uint8'))


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
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))

        r = np.zeros_like(img[center[0],...], dtype = 'int8')
        ax[0].imshow(np.dstack((img[center[0],...], img_mirror[center[0],...], r)))
        ax[0].set_title('Original')
        ax[0].axis('off')

        r = np.zeros_like(img[:,center[1],:], dtype = 'int8')
        ax[1].imshow(np.dstack((img[:,center[1],:], img_mirror[:,center[1],:], r)))
        ax[1].set_title('Mirror')
        ax[1].axis('off')

        r = np.zeros_like(img[...,center[2]], dtype = 'int8')
        ax[2].imshow(np.dstack((img[...,center[2]], img_mirror[...,center[2]], r)))
        ax[2].set_title('Optimized')
        ax[2].axis('off')