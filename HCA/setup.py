#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

requirements = [
    #'spinup @ https://github.com/openai/spinningup/archive/0.2.zip#egg=spinup-0.2',
    #'numpy',
    #'matplotlib',
    #'torch',
    #'gym'
]

extras_require = {
    'dev': ['pytest', 'flake8']
}

setup(
    name='hca',
    version='0.0.1',
    python_requires='>=3.6.1',
    description='Hindsight Credit Assignment',
    long_description='',
    author='Cathy Yeh',
    author_email='morgengruss@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require=extras_require,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            # TODO add stages
        ],
    },
    keywords='hca',
    classifiers=[
        'Development Status :: 3 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)
