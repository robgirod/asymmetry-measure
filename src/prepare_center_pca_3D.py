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
from scipy.ndimage import affine_transform


# ----------------- BEGIN CODE ------------------

def translate_center_of_mass(img):
    
    # First get the edges to speed up
    edges = sobel(img)
    otsu = threshold_otsu(edges)
    edges = edges > otsu

    # Center of image
    center = (img.shape[0]/2, img.shape[1]/2, img.shape[2]/2) 

    # Center of particle
    z, y, x = np.nonzero(img)
    center_x = x.mean() - center[2]
    center_y = y.mean() - center[1]
    center_z = z.mean() - center[0]

    tform = EuclideanTransform(translation = (center_z, center_y, center_x), dimensionality=3)

    return affine_transform(img, tform, order = 0, mode = 'constant')

def rotate_pca(img):

    # Find indices of the data points
    edges = sobel(img)
    otsu = threshold_otsu(edges)
    edges = edges > otsu
    z, y, x = np.nonzero(edges)

    # Center at (binary) center of mass
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    z_centered = z - z.mean()

    ZYX = np.vstack((z_centered, y_centered, x_centered)).T

    # Compute PCA to get major axes of the particle
    pca = decomposition.PCA(n_components = 3)
    pca.fit(ZYX)
    rot = pca.components_ # Retrieves the rotation matrix, this will work for the major axis on 0

    center = np.array(img.shape) // 2

    # Ensures rotation is centered
    tform1 = EuclideanTransform(translation = -center, dimensionality=3)

    # Rotation
    tform2 = EuclideanTransform(rotation = (0, 0, 0), dimensionality=3)
    tform2.params[:3, :3] = rot.T

    # Back to original coordinates
    tform3 = EuclideanTransform(translation = center, dimensionality=3)

    tform = tform1 + tform2 + tform3

    Q_aligned = affine_transform(img, tform, order = 0, mode = 'constant')

    # The rotation sometimes flips an axis, esp with det(rot) != 1, which is weird ...
    if np.linalg.det(rot.T) < 0:
        return np.flip(Q_aligned, axis = 2)
    else:
        return Q_aligned