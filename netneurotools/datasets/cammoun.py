# -*- coding: utf-8 -*-
"""
Loads parcellation information from Cammoun et al., 2012 (J Neurosci Methods)
"""

import pickle
from pkg_resources import resource_filename

_gcs_url = "https://github.com/LTS5/cmp/raw/master/cmp/data/colortable_and_gcs/my_atlas_gcs/myatlas{fname}"  # noqa
_gcs_files = [
    'P17_28_{hemi}.gcs', 'P1_16_{hemi}.gcs', 'P29_36_{hemi}.gcs',
    '_125_{hemi}.gcs', '_250_{hemi}.gcs', '_36_{hemi}.gcs', '_60_{hemi}.gcs'
]
_ctab_url = "https://raw.githubusercontent.com/LTS5/cmp/master/cmp/data/colortable_and_gcs/original_color_{fname}"  # noqa
_ctab_files = [
    '125_{hemi}.txt', '250_{hemi}.txt', '36_{hemi}.txt', '60_{hemi}.txt',
    'P17_28_{hemi}.txt', 'P1_16_{hemi}.txt', 'P29_36_{hemi}.txt'
]


def load_cammoun2012(scale):
    """
    Returns centroids / hemi assignment of parcels from Cammoun et al., 2012

    Centroids are defined on the spherical projection of the fsaverage cortical
    surface reconstruciton (FreeSurfer v6.0.1)

    Parameters
    ----------
    scale : {33, 60, 125, 250, 500}
        Scale of parcellation for which to get centroids / hemisphere
        assignments

    Returns
    -------
    centroids : (N, 3) numpy.ndarray
        Centroids of parcels defined by Cammoun et al., 2012 parcellation
    hemiid : (N,) numpy.ndarray
        Hemisphere assignment of `centroids`, where 0 indicates left and 1
        indicates right hemisphere

    References
    ----------
    Cammoun, L., Gigandet, X., Meskaldji, D., Thiran, J. P., Sporns, O., Do, K.
    Q., Maeder, P., and Meuli, R., & Hagmann, P. (2012). Mapping the human
    connectome at multiple scales with diffusion spectrum MRI. Journal of
    Neuroscience Methods, 203(2), 386-397.

    Examples
    --------
    >>> from netneurotools import datasets

    >>> coords, hemiid = datasets.load_cammoun2012(scale=33)
    >>> coords.shape, hemiid.shape
    ((68, 3), (68,))

    ``hemiid`` is a vector of 0 and 1 denoting which ``coords`` are in the
    left / right hemisphere, respectively:

    >>> hemiid
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1])
    """

    pckl = resource_filename('netneurotools', 'data/cammoun.pckl')

    if not isinstance(scale, int):
        try:
            scale = int(scale)
        except ValueError:
            raise ValueError('Provided `scale` must be integer in [33, 60, '
                             '125, 250, 500], not {}'.format(scale))
    if scale not in [33, 60, 125, 250, 500]:
        raise ValueError('Provided `scale` must be integer in [33, 60, 125, '
                         '250, 500], not {}'.format(scale))

    with open(pckl, 'rb') as src:
        data = pickle.load(src)['cammoun{}'.format(str(scale))]

    return data['centroids'], data['hemiid']
