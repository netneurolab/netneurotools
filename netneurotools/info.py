# -*- coding: utf-8 -*-

__version__ = '0.0.1'

NAME = 'netneurotools'
MAINTAINER = 'Network Neuroscience Lab'
EMAIL = 'rossmarkello@gmail.com'
VERSION = __version__
LICENSE = 'BSD-3'
DESCRIPTION = """\
Commonly used tools in the Network Neuroscience Lab\
"""
LONG_DESCRIPTION = 'README.rst'
LONG_DESCRIPTION_CONTENT_TYPE = 'text/x-rst'
URL = 'https://github.com/netneurolab/{name}'.format(name=NAME)
DOWNLOAD_URL = ('http://github.com/netneurolab/{name}/archive/{ver}.tar.gz'
                .format(name=NAME, ver=__version__))

INSTALL_REQUIRES = [
    'bctpy',
    'matplotlib',
    'numpy>=1.14',
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
    ],
    'tests': TESTS_REQUIRE
}

EXTRAS_REQUIRE['all'] = list(set([
    v for deps in EXTRAS_REQUIRE.values() for v in deps
]))

PACKAGE_DATA = {
    'netneurotools': ['tests/data/*', 'data/*']
}

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'
]
