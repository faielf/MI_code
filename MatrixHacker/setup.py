# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="matrixhacker",
    version="0.0.1",
    author="swolf",
    author_email="swolfforever@gmail.com",
    description="BCI algorithm implementations",
    long_description=long_description,
    long_description_content_type="text/markdown", 
    url="",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'scipy', 'scikit-learn', 'pandas', 'h5py', 'mne', 'torch', 'torchvision'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
