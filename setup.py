from setuptools import setup

setup(
    name="automl-prune",
    install_requires=[
        'torch',
        'torchvision',
        'torchtext',
        'sklearn',
        'tensorboardX',
        'termcolor',
    ]
)
