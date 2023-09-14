from setuptools import find_packages, setup

import os
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='uicsmodels',
    packages=find_packages(),
    install_required=required,
)
