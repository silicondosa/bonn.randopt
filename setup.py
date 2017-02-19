#!/usr/bin/env python

from setuptools import (
        setup as install,
        find_packages,
        )

VERSION = '0.0.1'

install(
        name='bonn',
        packages=['bonn'],
        version=VERSION,
        description='Bayesian optimization for randopt',
        author='Seb Arnold',
        author_email='smr.arnold@gmail.com',
        url = 'https://github.com/seba-1511/bonn.randopt',
        download_url = 'https://github.com/seba-1511/bonn.randopt/archive/0.0.1.zip',
        license='License :: OSI Approved :: Apache Software License',
        classifiers=[],
        scripts=[]
)
