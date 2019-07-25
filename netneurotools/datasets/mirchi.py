# -*- coding: utf-8 -*-
"""
Code for re-generating results from Mirchi et al., 2018 (SCAN)
"""

import os
from urllib.request import HTTPError, urlopen

import numpy as np

from .utils import _get_data_dir


TIMESERIES = ("https://s3.amazonaws.com/openneuro/ds000031/ds000031_R1.0.2"
              "/uncompressed/derivatives/sub-01/ses-{0}/"
              "sub-01_ses-{0}_task-rest_run-001_parcel-timeseries.txt")
BEHAVIOR = ("https://s3.amazonaws.com/openneuro/ds000031/ds000031_R1.0.4"
            "/uncompressed/sub-01/sub-01_sessions.tsv")
SESSIONS = [  # list of sessions with parcelled time series and all PANAS items
    '016', '019', '025', '026', '028', '029', '030', '032', '035', '037',
    '038', '039', '040', '041', '042', '043', '044', '045', '046', '047',
    '048', '049', '050', '051', '053', '054', '056', '057', '058', '059',
    '060', '061', '062', '063', '064', '065', '066', '067', '068', '069',
    '070', '071', '072', '073', '074', '075', '076', '077', '078', '079',
    '080', '081', '082', '083', '084', '085', '086', '087', '088', '089',
    '091', '092', '094', '095', '096', '097', '098', '099', '100', '101',
    '102', '103', '104'
]
PANAS = {  # specification for creation of PANAS subscales for item scores
    'negative': [
        'afraid', 'scared', 'nervous', 'jittery', 'irritable', 'hostile',
        'guilty', 'ashamed', 'upset', 'distressed'
    ],
    'positive': [
        'active', 'alert', 'attentive', 'determined', 'enthusiastic',
        'excited', 'inspired', 'interested', 'proud', 'strong'
    ],
    'fear': [
        'afraid', 'scared', 'frightened', 'nervous', 'jittery', 'shaky'
    ],
    'hostility': [
        'angry', 'hostile', 'irritable', 'scornful', 'disgusted', 'loathing'
    ],
    'guilt': [
        'guilty', 'ashamed', 'blameworthy', 'angry_at_self',
        'disgusted_with_self', 'dissatisfied_with_self'
    ],
    'sadness': [
        'sad', 'blue', 'downhearted', 'alone', 'lonely'
    ],
    'joviality': [
        'happy', 'joyful', 'delighted', 'cheerful', 'excited', 'enthusiastic',
        'lively', 'energetic',
    ],
    'self-assurance': [
        'proud', 'strong', 'confident', 'bold', 'daring', 'fearless'
    ],
    'attentiveness': [
        'alert', 'attentive', 'concentrating', 'determined'
    ],
    'shyness': [
        'shy', 'bashful', 'sheepish', 'timid'
    ],
    'fatigue': [
        'sleepy', 'tired', 'sluggish', 'drowsy'
    ],
    'serenity': [
        'calm', 'relaxed', 'at_ease'
    ],
    'surprise': [
        'amazed', 'surprised', 'astonished'
    ]
}


def _get_fc(data_dir=None, resume=True, verbose=1):
    """
    Gets functional connections from MyConnectome parcelled time series data

    Returns
    -------
    fc : (73, 198135) numpy.ndarray
        Functional connections (lower triangle)
    """

    # download time series data for all sessions
    ts = []
    for ses in SESSIONS:
        if verbose > 0:
            print('Fetching time series for session {}'.format(ses))
        out = urlopen(TIMESERIES.format(ses))
        if out.status == 200:
            ts.append(np.loadtxt(out.readlines()))
        else:
            raise HTTPError('Failed to fetch time series data: session {}'
                            .format(ses))

    # get upper triangle of correlation matrix for each session
    fc = [np.corrcoef(ses.T)[np.tril_indices(len(ses.T), k=-1)] for ses in ts]

    # return stacked sessions
    return np.row_stack(fc)


def _get_panas(data_dir=None, resume=True, verbose=1):
    """
    Gets PANAS subscales from MyConnectome behavioral data

    Returns
    -------
    panas : dict
        Where keys are PANAS subscales names and values are session-level
        composite measures
    """

    from numpy.lib.recfunctions import structured_to_unstructured as stu

    # download behavioral data
    out = urlopen(BEHAVIOR)
    if out.status == 200:
        data = out.readlines()
    else:
        raise HTTPError('Cannot fetch behavioral data')

    # drop sessions with missing PANAS items
    sessions = np.genfromtxt(data, delimiter='\t', usecols=0, dtype=object,
                             names=True, converters={0: lambda s: s.decode()})
    keeprows = np.isin(sessions, ['ses-{}'.format(f) for f in SESSIONS])
    panas = np.genfromtxt(data, delimiter='\t', names=True, dtype=float,
                          usecols=range(28, 91))[keeprows]

    # create subscales from individual item scores
    measures = {}
    for subscale, items in PANAS.items():
        measure = stu(panas[['panas{}'.format(f) for f in items]])
        measures[subscale] = measure.sum(axis=-1)

    return measures


def fetch_mirchi2018(data_dir=None, resume=True, verbose=1):
    """
    Downloads (and creates) dataset for replicating Mirchi et al., 2018, SCAN

    Parameters
    ----------
    data_dir : str, optional
        Directory to check for existing data files (if they exist) or to save
        generated data files. Files should be named mirchi2018_fc.npy and
        mirchi2018_panas.csv for the functional connectivity and behavioral
        data, respectively.

    Returns
    -------
    X : (73, 198135) numpy.ndarray
        Functional connections from MyConnectome rsfMRI time series data
    Y : (73, 13) numpy.ndarray
        PANAS subscales from MyConnectome behavioral data
    """

    data_dir = os.path.join(_get_data_dir(data_dir=data_dir), 'ds-mirchi2018')
    os.makedirs(data_dir, exist_ok=True)

    X_fname = os.path.join(data_dir, 'myconnectome_fc.npy')
    Y_fname = os.path.join(data_dir, 'myconnectome_panas.csv')

    if not os.path.exists(X_fname):
        X = _get_fc(data_dir=data_dir, resume=resume, verbose=verbose)
        np.save(X_fname, X, allow_pickle=False)
    else:
        X = np.load(X_fname, allow_pickle=False)

    if not os.path.exists(Y_fname):
        Y = _get_panas(data_dir=data_dir, resume=resume, verbose=verbose)
        np.savetxt(Y_fname, np.column_stack(list(Y.values())),
                   header=','.join(Y.keys()), delimiter=',', fmt='%i')
        # convert dictionary to structured array before returning
        Y = np.array([tuple(row) for row in np.column_stack(list(Y.values()))],
                     dtype=dict(names=list(Y.keys()), formats=['i8'] * len(Y)))
    else:
        Y = np.genfromtxt(Y_fname, delimiter=',', names=True, dtype=int)

    return X, Y
