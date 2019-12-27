#Adapted from https://github.com/Qiskit/qiskit-aer/blob/master/setup.py

import os
from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

requirements = [
    "numpy>=1.13"
]

# Handle version.
VERSION = "0.3.0"

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'README.md')
with open(README_PATH) as readme_file:
    README = readme_file.read()

class get_opencl_library(object):
    """Helper class to determine if OpenCL is present"""

    def __init__(self):
        pass

    def __str__(self):
        from ctypes import cdll
        try:
            cdll.LoadLibrary('libOpenCL.so.1')
            return 'OpenCL'
        except OSError:
            # Return something inoffensive that always works
            return 'm'

qrack_extension = [Extension(
                       'qiskit.providers.qrack.backends.qrack_controller_wrapper',
                       ['qiskit/providers/qrack/backends/wrappers/qrack_controller_wrapper.pyx'],
                       include_dirs=[],
                       libraries=['qrack', str(get_opencl_library())],
                       language="c++"
                  )]

setup(
    name='qiskit-qrack-provider',
    version=VERSION,
    ext_modules = cythonize(qrack_extension),
    packages=['qiskit.providers.qrack',
              'qiskit.providers.qrack.backends'],
    description="Qiskit Qrack Provider - Qrack High-Performance GPU simulation for Qiskit",
    long_description=README,
    long_description_content_type='text/markdown',
    url="https://github.com/vm6502q/qiskit-qrack-provider",
    author="Daniel Strano",
    author_email="stranoj@gmail.com",
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
    keywords="qiskit qrack simulator quantum addon backend",
    install_requires=requirements,
    setup_requires=['Cython'],
    include_package_data=True,
    zip_safe=False
)
