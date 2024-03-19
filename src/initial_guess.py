'''
Functions to register a mirror image with Dipy, so as to place it a global minimum for the Hausdorff distance
'''

from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi
from dipy.data.fetcher import fetch_syn_data
from dipy.io.image import load_nifti
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform2D,
                                   RigidTransform2D,
                                   AffineTransform2D)

def dipy_translate(static, moving, nbins = 32, sampling_prop = None):
 
    metric = MutualInformationMetric(nbins, sampling_prop)

    level_iters = [10000, 1000, 100]
    sigmas = [2.0, 1.0, 0.0]
    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform2D()
    params0 = None
    starting_affine = None
    translation = affreg.optimize(static, moving, transform, params0,
                            None, None,
                            starting_affine=starting_affine)

    return translation.transform(moving), translation

def dipy_rigid(static, moving, translation):

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    level_iters = [10000, 1000, 100]
    sigmas = [2.0, 1.0, 0.0]
    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = RigidTransform2D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0,
                            None, None,
                            starting_affine=starting_affine)

    return rigid.transform(moving), rigid

def prealign(img, mirror):

    mirror_translate, transform = dipy_translate(img, mirror)
    mirror_rigid, transform = dipy_rigid(img, mirror_translate, transform)

    return mirror_rigid, transform