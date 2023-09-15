from setuptools import find_packages, setup

setup(
    name='uicsmodels',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'blackjax @ git+https://github.com/Hesterhuijsdens/blackjax.git',
        'jaxkern @ git+https://github.com/JaxGaussianProcesses/JaxKern.git',
        'distrax==0.1.4'
    ]
)
