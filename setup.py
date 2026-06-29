"""Set up file for bqskit-shuttling."""
from setuptools import setup, find_namespace_packages

setup(
    name='bqskit-shuttling',
    description='BQSKit extension for compiling to shuttling ion-trap architectures.',
    version='0.1.0',
    packages=find_namespace_packages(),
    include_package_data=True,
    package_data={
        'bqskit.shuttling.qccd': ['benchmark_circuits/*.qasm'],
    },
    install_requires=['bqskit', 'numpy', 'rustworkx'],
    python_requires='>=3.8, <4.0'
)
