"""
Functions to prepare a 2D image for Hausdorff chirality calculation

center to center of mass
align axes with PCA

"""

from skimage import registration
from skimage.transform import warp, downscale_local_mean, EuclideanTransform, rotate
from skimage import img_as_float32
from skimage.exposure import rescale_intensity
import mrcfile
from skimage.metrics import hausdorff_distance, hausdorff_pair
from skimage.filters import threshold_otsu, sobel
from skimage.draw import line_nd, ellipse, polygon
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

    # Find indices of the data points
    edges = sobel(img)
    otsu = threshold_otsu(edges)
    edges = edges > otsu
    y, x = np.nonzero(edges)

    # Center at center of mass
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    YX = np.vstack((y_centered, x_centered)).T

    # Compute PCA to get axes of inertia
    pca = decomposition.PCA(n_components = 2)
    pca.fit(YX)
    rot = pca.components_ # Retrieves the rotation matrix

    # Flip Rotation matrix PC1 and PC2 if det is -1
    # if round(np.linalg.det(rot)) == -1:
    #   rot = np.flipud(rot)


    # Create transformation matrix
    # First, center the rotation
    # to ensure that the image is rotated around the center of the image/volume
    rows, cols = img.shape[0], img.shape[1]
    center = np.array((cols, rows)) / 2. - 0.5

    tform1 = EuclideanTransform(translation = center)

    # Rotation
    tform2 = EuclideanTransform(rotation = 0)
    tform2.params[:2, :2] = rot

    # Back to original position
    tform3 = EuclideanTransform(translation = -center)

    # Perform the transforms
    tform = tform3 + tform2 + tform1
    return warp(img, tform, order = 0, mode = 'constant'), rot