"""Functions for handling datasets."""


from .fetch_template import (
    fetch_fsaverage, fetch_hcp_standards, fetch_civet,
    fetch_conte69, fetch_yerkes19
)


from .fetch_atlas import (
    # cortical
    fetch_cammoun2012, fetch_schaefer2018, fetch_mmpall,
    # subcortical
    fetch_pauli2018, fetch_ye2020,
    # annotation
    fetch_voneconomo
)


from .fetch_project import (
    # old projects
    fetch_vazquez_rodriguez2019, fetch_mirchi2018,
    # new projects
    fetch_hansen_manynetworks, fetch_hansen_receptors,
    fetch_hansen_genecognition, fetch_hansen_brainstem,
    fetch_shafiei_megfmrimapping, fetch_shafiei_megdynamics,
    fetch_suarez_mami,
    # example data
    fetch_famous_gmat,
    # resources
    fetch_neurosynth
)

from .datasets_utils import (
    FREESURFER_IGNORE, _get_freesurfer_subjid
)


__all__ = [
    # fetch_template
    'fetch_fsaverage', 'fetch_hcp_standards', 'fetch_civet',
    'fetch_conte69', 'fetch_yerkes19',
    # fetch_atlas
    'fetch_cammoun2012', 'fetch_schaefer2018', 'fetch_mmpall',
    'fetch_pauli2018', 'fetch_ye2020',
    'fetch_voneconomo',
    # fetch_project
    'fetch_vazquez_rodriguez2019', 'fetch_mirchi2018',
    'fetch_hansen_manynetworks', 'fetch_hansen_receptors',
    'fetch_hansen_genecognition', 'fetch_hansen_brainstem',
    'fetch_shafiei_megfmrimapping', 'fetch_shafiei_megdynamics',
    'fetch_suarez_mami',
    'fetch_famous_gmat',
    'fetch_neurosynth',
    # datasets_utils
    'FREESURFER_IGNORE', '_get_freesurfer_subjid'
]
