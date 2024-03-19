"""
Functions to prepare a 2D image for Hausdorff chirality calculation

center to center of mass
align axes with PCA

"""

from skimage import registration
from skimage.transform import warp, downscale_local_mean, EuclideanTransform, affine_transform
from skimage import img_as_float32
from skimage.exposure import rescale_intensity
import mrcfile
from skimage.metrics import hausdorff_distance, hausdorff_pair
from skimage.filters import threshold_otsu, sobel
from skimage.draw import line_nd, ellipse, polygon
from scipy.ndimage import affine_transform
import numpy as np
from sklearn import decomposition

# ----------------- BEGIN CODE ------------------

def translate_center_of_mass(img):
    
    # First get the edges to speed up
    edges = sobel(img)
    otsu = threshold_otsu(edges)
    edges = edges > otsu

    # Center of image
    center = img.shape//2

    # Center of particle
    z, y, x = np.nonzero(edges)
    center_x = x.mean() - center[0]
    center_y = y.mean() - center[0]
    tform = EuclideanTransform(translation = (center_x, center_y))
    
    return warp(img, tform, order = 0, mode = 'wrap')

def rotate_pca(img):

    # Reduce size if needed, here if any axis is > 150 voxels
    if np.any(np.array(img.shape) > 150):
        bin_factor = np.ceil(max(img.shape) / 150)
        img = downscale_local_mean(img, bin_factor)

        print(f'Reduced the image size to {img.shape} for speed')

    # Find indices of the data points
    edges = sobel(img)
    otsu = threshold_otsu(edges)
    edges = edges > otsu
    z, y, x = np.nonzero(edges)

    # Center at center of mass
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    z_centered = z - z.mean()

    ZYX = np.vstack((z_centered, y_centered, x_centered)).T

    # Compute PCA to get axes of inertia
    pca = decomposition.PCA(n_components = 3)
    pca.fit(ZYX)
    rot = pca.components_ # Retrieves the rotation matrix

    slices, rows, cols = img.shape[0], img.shape[1], img.shape[2] 
    center = np.array((slices, cols, rows)) / 2. - 0.5

    tform1 = EuclideanTransform(translation = center, dimensionality=3)

    # Rotation
    tform2 = EuclideanTransform(rotation = (0, 0, 0), dimensionality=3)
    tform2.params[:3, :3] = rot.T

    # Back to original position
    tform3 = EuclideanTransform(translation = -center, dimensionality=3)

    # Perform the transforms
    tform = tform3 + tform2 + tform1
    return affine_transform(img, tform, order = 0, mode = 'constant'), rot.T