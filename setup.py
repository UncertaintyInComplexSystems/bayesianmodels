from setuptools import find_packages, setup

setup(
    name='uicsmodels',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'jaxtyping',
        'distrax==0.1.4',
        'jaxkern @ git+https://github.com/JaxGaussianProcesses/JaxKern.git',
        'blackjax'
    ]
)
