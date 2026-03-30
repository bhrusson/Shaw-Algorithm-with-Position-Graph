"""Set up file for bqskit-shuttling."""
from setuptools import setup, find_namespace_packages

setup(
    name='bqskit-shuttling',
    description='BQSKit extension for compiling to shuttling ion-trap architectures.',
    version='0.1.0',
    packages=find_namespace_packages(),
    install_requires=['bqskit', 'numpy'],
    python_requires='>=3.8, <4.0'
)