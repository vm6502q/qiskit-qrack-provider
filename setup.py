#Adapted from https://github.com/Qiskit/qiskit-aer/blob/master/setup.py

import os
from distutils.core import setup

requirements = [
    'numpy>=1.16.3',
    'pyqrack>=1.21.0'
]

# Handle version.
VERSION = "0.8.1"

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'README.md')
with open(README_PATH) as readme_file:
    README = readme_file.read()

setup(
    name='qiskit-qrack-provider',
    version=VERSION,
    packages=['qiskit.providers.qrack',
              'qiskit.providers.qrack.backends'],
    description="Qiskit Qrack Provider - Qrack High-Performance GPU simulation for Qiskit",
    long_description=README,
    long_description_content_type='text/markdown',
    url="https://github.com/vm6502q/qiskit-qrack-provider",
    author="Daniel Strano",
    author_email="dan@unitary.fund",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit qrack pyqrack simulator quantum addon backend",
    install_requires=requirements
)
