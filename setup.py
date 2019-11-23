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

def find_qiskit_packages():
    location = 'qiskit/providers'
    prefix = 'qiskit.providers'
    qrack_packages = find_packages(where=location)
    pkg_list = list(
        map(lambda package_name: '{}.{}'.format(prefix, package_name),
            qrack_packages)
    )
    return pkg_list


setup(
    name='qiskit-qrack-provider',
    version="0.2.0",
    packages=find_qiskit_packages() if not dummy_install else [],
    cmake_source_dir='.',
    description="Qiskit Qrack Provider - Qrack High-Performance GPU simulation for Qiskit",
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
        "Topic :: Scientific/Engineering",
    ],
    install_requires=requirements,
    include_package_data=True,
    cmake_args=["-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.9"],
    keywords="qiskit qrack simulator quantum addon backend"
)
