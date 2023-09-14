from setuptools import find_packages, setup

setup(
    name='uicsmodels',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'jax',
        'jaxtyping',
        'distrax @ git+https://github.com/deepmind/distrax.git@f6e656c',
        'jaxkern @ git+https://github.com/JaxGaussianProcesses/JaxKern.git',
        'blackjax @ git+https://github.com/Hesterhuijsdens/blackjax.git'
    ]
)
