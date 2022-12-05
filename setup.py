#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='transformiloop',
      version='0.0.2',
      description='Transformer model to be used to train and infer portiloop EEG data',
      author='Milo Sobral',
      author_email='milosobral@gmail.com',
      url='https://github.com/milo-sobral/portiloop_transformer',
      packages=find_packages(include=['transformiloop']),
      install_requires=[
        'numpy',
        'matplotlib',
        'torch',
        'einops',
        'wandb',
        'torchinfo',
        'scikit-learn',
        'pandas',
        'torchmetrics',
        'py3nvml',
        'tlspyo',
        'pyedflib']
     )