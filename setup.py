# -*- coding: utf-8 -*-


def main():
    import versioneer
    from io import open
    import os.path as op
    from inspect import getfile, currentframe
    from setuptools import setup, find_packages

    root_dir = op.dirname(op.abspath(getfile(currentframe())))

    ldict = locals()
    with open(op.join(root_dir, 'netneurotools', 'info.py')) as infofile:
        exec(infofile.read(), globals(), ldict)

    with open(op.join(root_dir, ldict['__longdesc__'])) as src:
        ldict['__longdesc__'] = src.read()

    ldict.setdefault('__version__', versioneer.get_version())
    ldict.setdefault('__cmdclass__', versioneer.get_cmdclass())
    DOWNLOAD_URL = (
        'https://github.com/{author}/{name}/archive/{ver}.tar.gz'.format(
            author=ldict['__author__'],
            name=ldict['__packagename__'],
            ver=ldict['__version__']))

    setup(
        name=ldict['__packagename__'],
        version=ldict['__version__'],
        description=ldict['__description__'],
        long_description=ldict['__longdesc__'],
        long_description_content_type=ldict['__longdesctype__'],
        author=ldict['__author__'],
        author_email=ldict['__email__'],
        maintainer=ldict['__maintainer__'],
        maintainer_email=ldict['__email__'],
        url=ldict['__url__'],
        license=ldict['__license__'],
        classifiers=ldict['CLASSIFIERS'],
        download_url=DOWNLOAD_URL,
        install_requires=ldict['INSTALL_REQUIRES'],
        packages=find_packages(exclude=['pyls/tests']),
        package_data=ldict['PACKAGE_DATA'],
        tests_require=ldict['TESTS_REQUIRE'],
        extras_require=ldict['EXTRAS_REQUIRE'],
        cmdclass=ldict['__cmdclass__']
    )


if __name__ == '__main__':
    main()
