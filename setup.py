#!/usr/bin/env python
"""omf: API Library for Open Mining Format"""

from distutils.core import setup
from setuptools import find_packages

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Natural Language :: English",
]

with open("README.rst") as f:
    LONG_DESCRIPTION = "".join(f.readlines())

setup(
    name="omf",
    version="2.0.0a0",
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "numpy>=1.7",
        "properties==0.6.0b0",
        "pypng",
        "vectormath>=0.2.0",
    ],
    author="Global Mining Guidelines Group",
    author_email="info@gmggroup.org",
    description="API Library for Open Mining Format",
    long_description=LONG_DESCRIPTION,
    keywords="mining data interchange",
    url="https://gmggroup.org",
    download_url="https://github.com/gmggroup/omf",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    license="MIT License",
    use_2to3=False,
)
