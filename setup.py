# MINIMAL setup.py file for a Python module

from setuptools import setup, find_packages

setup(
    name='neuroparc',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'nibabel',
        'numpy',
        'nilearn',
    ],
)