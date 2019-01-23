# -*- coding: utf-8 -*-

import os
import sys


def main():
    from setuptools import setup, find_packages

    if sys.version_info < (3, 5):
        raise SystemError('You need Python version 3.5 or above to use '
                          + 'this package. Current\nversion detected is '
                          + sys.version.split(' ')[0])

    ldict = locals()
    curr_path = os.path.dirname(__file__)
    with open(os.path.join(curr_path, 'netneurotools', 'info.py')) as infofile:
        exec(infofile.read(), globals(), ldict)

    with open(os.path.join(curr_path, ldict['LONG_DESCRIPTION'])) as src:
        ldict['LONG_DESCRPITION'] = src.read()

    setup(
        classifiers=ldict['CLASSIFIERS'],
        description=ldict['DESCRIPTION'],
        download_url=ldict['DOWNLOAD_URL'],
        extras_require=ldict['EXTRAS_REQUIRE'],
        install_requires=ldict['INSTALL_REQUIRES'],
        license=ldict['LICENSE'],
        long_description=ldict['LONG_DESCRIPTION'],
        long_description_content_type=ldict['LONG_DESCRIPTION_CONTENT_TYPE'],
        maintainer=ldict['MAINTAINER'],
        maintainer_email=ldict['EMAIL'],
        name=ldict['NAME'],
        packages=find_packages(exclude=['netneurotools/tests']),
        package_data=ldict['PACKAGE_DATA'],
        tests_require=ldict['TESTS_REQUIRE'],
        url=ldict['URL'],
        version=ldict['VERSION'],
    )


if __name__ == '__main__':
    main()
