# -*- coding: utf-8 -*-
"""
Code for re-generating results from Mirchi et al., 2018 (SCAN)
"""

from io import StringIO
import os

import numpy as np
import pandas as pd
import requests

from .utils import _get_dataset_dir


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


def _get_fc():
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
        out = requests.get(TIMESERIES.format(ses))
        if out.status_code == 200:
            ts.append(np.loadtxt(StringIO(out.text)))

    # get upper triangle of correlation matrix for each session
    fc = [np.corrcoef(ses.T)[np.tril_indices(len(ses.T), k=-1)] for ses in ts]

    # return stacked sessions
    return np.row_stack(fc)


def _get_panas():
    """
    Gets PANAS subscales from MyConnectome behavioral data

    Returns
    -------
    panas : (73, 13) pandas.DataFrame
        PANAS subscales
    """

    # download behavioral data
    out = requests.get(BEHAVIOR)
    if out.status_code == 200:
        behavior = pd.read_csv(StringIO(out.text), sep='\t')
        behavior = behavior.set_index('sescode')
    else:
        raise requests.ConnectionError('Cannot get behavioral data')

    # drop sessions with missing PANAS items / time series data
    ses = ['ses-{}'.format(f) for f in SESSIONS]
    cols = [f for f in behavior.columns if f.startswith('panas')]
    behavior = behavior.dropna(how='any', axis=0, subset=cols)
    behavior = behavior.loc[ses, cols]

    # create subscales from individual item scores
    panas = pd.DataFrame()
    for subscale, items in PANAS.items():
        measure = behavior[['panas:{}'.format(f) for f in items]].sum(axis=1)
        panas[subscale] = measure

    # return z-scored data
    return (panas - panas.mean(axis=0)) / panas.std(axis=0, ddof=1)


def fetch_mirchi2018(data_dir=None):
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
    X : (73, 198135) pandas.DataFrame
        Functional connections from MyConnectome rsfMRI time series data
    Y : (73, 13) pandas.DataFrame
        PANAS subscales from MyConnectome behavioral data
    """

    data_dir = _get_dataset_dir('mirchi2018', data_dir=data_dir)
    os.makedirs(data_dir, exist_ok=True)

    X_fname = os.path.join(data_dir, 'myconnectome_fc.npy')
    Y_fname = os.path.join(data_dir, 'myconnectome_panas.csv')

    if not os.path.exists(X_fname):
        X = _get_fc()
        np.save(X_fname, X)
    else:
        X = np.load(X_fname)

    if not os.path.exists(Y_fname):
        Y = _get_panas()
        Y.to_csv(Y_fname)
    else:
        Y = pd.read_csv(Y_fname)

    return X, Y
