#!/usr/bin/env python

from setuptools import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="MoA",
    version="0.1.1",
    packages=setuptools.find_packages(),
    license="Apache-2.0",
    long_description=long_description,
    # install_requires=["networkx"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
