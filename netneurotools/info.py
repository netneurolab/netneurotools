# -*- coding: utf-8 -*-

__author__ = 'netneurolab'
__copyright__ = 'Copyright 2018, netneurotools developers'
__credits__ = ['Ross Markello', 'Bratislav Misic', 'Golia Shafiei']
__license__ = 'BSD-3'
__maintainer__ = 'Network Neuroscience Lab'
__email__ = 'rossmarkello@gmail.com'
__status__ = 'Prototype'
__url__ = 'https://github.com/netneurolab/netneurotools'
__packagename__ = 'netneurotools'
__description__ = """\
Commonly used tools in the Network Neuroscience Lab\
"""
__longdesc__ = 'README.rst'
__longdesctype__ = 'text/x-rst'

INSTALL_REQUIRES = [
    'bctpy',
    'matplotlib',
    'nibabel',
    'numpy>=1.14',
    'pandas',
    'requests',
    'scikit-learn',
    'scipy',
    'seaborn'
]

TESTS_REQUIRE = [
    'codecov',
    'pytest',
    'pytest-cov'
]

EXTRAS_REQUIRE = {
    'doc': [
        'sphinx>=1.2',
        'sphinx_rtd_theme',
    ],
    'numba': [
        'numba',
    ],
    'plotting': [
        'mayavi',
        'pysurfer'
    ],
    'tests': TESTS_REQUIRE
}

EXTRAS_REQUIRE['all'] = list(set([
    v for deps in EXTRAS_REQUIRE.values() for v in deps
]))

PACKAGE_DATA = {
    'netneurotools': [
        'tests/data/*', 'data/*'
    ]
}

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'
]
