#Adapted from https://github.com/Qiskit/qiskit-aer/blob/master/setup.py

import os
try:
    from skbuild import setup
    dummy_install = False
except:
    print(""" WARNING
              =======
              scikit-build package is needed to build Qrack sources.
              Please, install scikit-build and reinstall""")
    from setuptools import setup
    dummy_install = True
from setuptools import find_packages

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

setup(
    name='qiskit-qrack-provider',
    version=VERSION,
    packages=['qiskit.providers.qrack',
              'qiskit.providers.qrack.backends',
              'qiskit.providers.qrack.backends.qrack_controller_wrapper'],
    package_dir={'qiskit.providers.qrack.backends.qrack_controller_wrapper':
                 'qiskit/providers/qrack/backends'},
    package_data={'qiskit.providers.qrack.backends.qrack_controller_wrapper':
                  ['qrack_controller_wrapper.*.so']},
    cmake_source_dir='.',
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
    setup_requires=['scikit-build', 'cmake', 'Cython'],
    include_package_data=True,
    cmake_args=["-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.9"],
    zip_safe=False
)
