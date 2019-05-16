# (C) Copyright Daniel Strano and the Qrack contributors, 2019
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Cython wrapper for QrackController.
"""

import numpy as np
from cpython cimport array
from libcpp.vector cimport vector

cdef extern from "qrack_controller.hpp" namespace "AER::Simulator":
    cdef cppclass QrackController:
        QrackController() except +
        void initialize_qreg(unsigned char) except +
        #initialize_qreg(uint64_t, const vector[complex]) except +
        void u(unsigned char target, double* params)
        void u2(unsigned char target, double* params)
        void u1(unsigned char target, double* params)
        void cx(unsigned char* bits, unsigned char ctrlCount)
        void cz(unsigned char* bits, unsigned char ctrlCount)
        void h(unsigned char target)
        void x(unsigned char target)
        void y(unsigned char target)
        void z(unsigned char target)
        void s(unsigned char target)
        void sdg(unsigned char target)
        void t(unsigned char target)
        void tdg(unsigned char target)
        void swap(unsigned char target1, unsigned char target2)
        void reset(unsigned char target)
        vector[double complex] amplitudes()
        vector[double] probabilities()
        int measure_all()


cdef class PyQrackController:
    cdef QrackController *c_class

    def __cinit__(self):
        # Set pointer to null on object init
        self.c_class = NULL

    def __dealoc__(self):
        if self.c_class is not NULL:
            del self.c_class

    def initialize_qreg(self, num_qubits):
        self.c_class.initialize_qreg(num_qubits)

    def u(self, target, params):
        cdef array.array params_array = array.array('d', params)
        self.c_class.u(target, params_array.data.as_doubles)

    def u2(self, target, params):
        cdef array.array params_array = array.array('d', params)
        self.c_class.u2(target, params_array.data.as_doubles)

    def u1(self, target, params):
        cdef array.array params_array = array.array('d', params)
        self.c_class.u1(target, params_array.data.as_doubles)

    def cx(self, bits, ctrlCount):
        cdef array.array bits_array = array.array('i', bits)
        self.c_class.cx(bits_array.data.as_uchars, ctrlCount)

    def cz(self, bits, ctrlCount):
        cdef array.array bits_array = array.array('i', bits)
        self.c_class.cz(bits_array.data.as_uchars, ctrlCount)

    def h(self, target):
        self.c_class.h(target)

    def x(self, target):
        self.c_class.x(target)

    def y(self, target):
        self.c_class.y(target)

    def z(self, target):
        self.c_class.z(target)

    def s(self, target):
        self.c_class.s(target)

    def sdg(self, target):
        self.c_class.sdg(target)

    def t(self, target):
        self.c_class.t(target)

    def tdg(self, target):
        self.c_class.t(target)

    def swap(self, target1, target2):
        self.c_class.swap(target1, target2)

    def reset(self, target):
        self.c_class.reset(target)

    def amplitudes(self):
        return np.array(self.c_class.amplitudes())

    def probabilities(self):
        return np.array(self.c_class.probabilities())

    def measure_all(self):
        return self.c_class.measure_all()


def qrack_controller_factory():
    """Expose QrackController to Python"""
    cdef PyQrackController py_obj = PyQrackController()
    # Set extension pointer to existing C++ class ptr
    py_obj.c_class = new QrackController()
    return py_obj
