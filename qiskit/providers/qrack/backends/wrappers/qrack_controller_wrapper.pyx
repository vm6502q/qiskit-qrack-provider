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

from collections.abc import Iterable
import numpy as np
from cpython cimport array
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.vector cimport vector

cdef extern from "qrack_controller.hpp" namespace "AER::Simulator":
    cdef cppclass QrackController:
        QrackController() except +
        void initialize_qreg(bool use_opencl, bool use_qunit, bool use_qpager, bool use_stabilizer, unsigned char num_qubits, int opencl_device_id, bool opencl_multi, bool doNormalize, double zero_threshold) except +
        QrackController* clone()
        void u(unsigned char* bits, unsigned char bitCount, double* params)
        void u2(unsigned char* bits, unsigned char bitCount, double* params)
        void u1(unsigned char* bits, unsigned char bitCount, double* params)
        void unitary1qb(unsigned char* bits, unsigned char bitCount, double* params)
        void cu(unsigned char* bits, unsigned char ctrlCount, double* params)
        void cu2(unsigned char* bits, unsigned char ctrlCount, double* params)
        void cu1(unsigned char* bits, unsigned char ctrlCount, double* params)
        void cx(unsigned char* bits, unsigned char ctrlCount)
        void cz(unsigned char* bits, unsigned char ctrlCount)
        void ch(unsigned char* bits, unsigned char ctrlCount)
        void h(unsigned char* bits, unsigned char bitCount)
        void x(unsigned char* bits, unsigned char bitCount)
        void sx(unsigned char* bits, unsigned char bitCount)
        void y(unsigned char* bits, unsigned char bitCount)
        void z(unsigned char* bits, unsigned char bitCount)
        void s(unsigned char* bits, unsigned char bitCount)
        void sdg(unsigned char* bits, unsigned char bitCount)
        void t(unsigned char* bits, unsigned char bitCount)
        void tdg(unsigned char* bits, unsigned char bitCount)
        void rx(unsigned char* bits, unsigned char bitCount, double* params)
        void ry(unsigned char* bits, unsigned char bitCount, double* params)
        void rz(unsigned char* bits, unsigned char bitCount, double* params)
        void swap(unsigned char target1, unsigned char target2)
        void cswap(unsigned char* bits, unsigned char ctrlCount)
        void initialize(unsigned char* bits, unsigned char bitCount, double* params)
        void multiplexer(unsigned char* bits, unsigned char ctrlCount, double* params)
        void reset(unsigned char target)
        vector[double complex] amplitudes()
        vector[double] probabilities()
        unsigned long int measure(unsigned char* bits, unsigned char bitCount)
        unsigned long int measure_all()
        map[unsigned long int, int] measure_shots(unsigned char* bits, unsigned char bitCount, unsigned int shots)

cdef class PyQrackController:
    cdef QrackController *c_class

    def __cinit__(self):
        # Set pointer to null on object init
        self.c_class = NULL

    def __dealoc__(self):
        if self.c_class is not NULL:
            del self.c_class

    def _complex_cast(self, x):
        if isinstance(x, complex):
            return [x.real, x.imag]
        else:
            return [x, 0.0]

    def initialize_qreg(self, use_opencl, use_qunit, use_qpager, use_stabilizer, num_qubits, opencl_device_id, opencl_multi, doNormalize, zero_threshold):
        self.c_class.initialize_qreg(use_opencl, use_qunit, use_qpager, use_stabilizer, num_qubits, opencl_device_id, opencl_multi, doNormalize, zero_threshold)

    def clone(self):
        cdef PyQrackController py_obj = PyQrackController()
        # Set extension pointer to clone of existing C++ class ptr
        py_obj.c_class = self.c_class.clone()
        return py_obj

    def u(self, bits, params):
        bitCount = len(bits)
        cdef array.array bits_array = array.array('B', bits)
        cdef array.array params_array = array.array('d', params)
        self.c_class.u(bits_array.data.as_uchars, bitCount, params_array.data.as_doubles)

    def u2(self, bits, params):
        bitCount = len(bits)
        cdef array.array bits_array = array.array('B', bits)
        cdef array.array params_array = array.array('d', params)
        self.c_class.u2(bits_array.data.as_uchars, bitCount, params_array.data.as_doubles)

    def u1(self, bits, params):
        bitCount = len(bits)
        cdef array.array bits_array = array.array('B', bits)
        cdef array.array params_array = array.array('d', params)
        self.c_class.u1(bits_array.data.as_uchars, bitCount, params_array.data.as_doubles)

    def unitary1qb(self, bits, params):
        bitCount = len(bits)
        cdef array.array bits_array = array.array('B', bits)

        items = [item for sublist in params for item in sublist] #rows
        items = [item for sublist in items for item in self._complex_cast(sublist)] #components
        cdef array.array params_array = array.array('d', items)

        self.c_class.unitary1qb(bits_array.data.as_uchars, bitCount, params_array.data.as_doubles)

    def cu(self, bits, params):
        ctrlCount = len(bits) - 1
        cdef array.array bits_array = array.array('B', bits)
        cdef array.array params_array = array.array('d', params)
        self.c_class.cu(bits_array.data.as_uchars, ctrlCount, params_array.data.as_doubles)

    def cu2(self, bits, params):
        ctrlCount = len(bits) - 1
        cdef array.array bits_array = array.array('B', bits)
        cdef array.array params_array = array.array('d', params)
        self.c_class.cu2(bits_array.data.as_uchars, ctrlCount, params_array.data.as_doubles)

    def cu1(self, bits, params):
        ctrlCount = len(bits) - 1
        cdef array.array bits_array = array.array('B', bits)
        cdef array.array params_array = array.array('d', params)
        self.c_class.cu1(bits_array.data.as_uchars, ctrlCount, params_array.data.as_doubles)

    def cx(self, bits):
        ctrlCount = len(bits) - 1
        cdef array.array bits_array = array.array('B', bits)
        self.c_class.cx(bits_array.data.as_uchars, ctrlCount)

    def cz(self, bits):
        ctrlCount = len(bits) - 1
        cdef array.array bits_array = array.array('B', bits)
        self.c_class.cz(bits_array.data.as_uchars, ctrlCount)

    def ch(self, bits):
        ctrlCount = len(bits) - 1
        cdef array.array bits_array = array.array('B', bits)
        self.c_class.ch(bits_array.data.as_uchars, ctrlCount)

    def h(self, bits):
        bitCount = len(bits)
        cdef array.array bits_array = array.array('B', bits)
        self.c_class.h(bits_array.data.as_uchars, bitCount)

    def x(self, bits):
        bitCount = len(bits)
        cdef array.array bits_array = array.array('B', bits)
        self.c_class.x(bits_array.data.as_uchars, bitCount)
        
    def sx(self, bits):
        bitCount = len(bits)
        cdef array.array bits_array = array.array('B', bits)
        self.c_class.sx(bits_array.data.as_uchars, bitCount)

    def y(self, bits):
        bitCount = len(bits)
        cdef array.array bits_array = array.array('B', bits)
        self.c_class.y(bits_array.data.as_uchars, bitCount)

    def z(self, bits):
        bitCount = len(bits)
        cdef array.array bits_array = array.array('B', bits)
        self.c_class.z(bits_array.data.as_uchars, bitCount)

    def s(self, bits):
        bitCount = len(bits)
        cdef array.array bits_array = array.array('B', bits)
        self.c_class.s(bits_array.data.as_uchars, bitCount)

    def sdg(self, bits):
        bitCount = len(bits)
        cdef array.array bits_array = array.array('B', bits)
        self.c_class.sdg(bits_array.data.as_uchars, bitCount)

    def t(self, bits):
        bitCount = len(bits)
        cdef array.array bits_array = array.array('B', bits)
        self.c_class.t(bits_array.data.as_uchars, bitCount)

    def tdg(self, bits):
        bitCount = len(bits)
        cdef array.array bits_array = array.array('B', bits)
        self.c_class.tdg(bits_array.data.as_uchars, bitCount)

    def rx(self, bits, params):
        bitCount = len(bits)
        cdef array.array bits_array = array.array('B', bits)
        cdef array.array params_array = array.array('d', params)
        self.c_class.rx(bits_array.data.as_uchars, bitCount, params_array.data.as_doubles)

    def ry(self, bits, params):
        bitCount = len(bits)
        cdef array.array bits_array = array.array('B', bits)
        cdef array.array params_array = array.array('d', params)
        self.c_class.ry(bits_array.data.as_uchars, bitCount, params_array.data.as_doubles)

    def rz(self, bits, params):
        bitCount = len(bits)
        cdef array.array bits_array = array.array('B', bits)
        cdef array.array params_array = array.array('d', params)
        self.c_class.rz(bits_array.data.as_uchars, bitCount, params_array.data.as_doubles)

    def swap(self, target1, target2):
        self.c_class.swap(target1, target2)

    def cswap(self, bits):
        ctrlCount = len(bits) - 2
        cdef array.array bits_array = array.array('B', bits)
        self.c_class.cswap(bits_array.data.as_uchars, ctrlCount)

    def initialize(self, bits, params):
        bitCount = len(bits)
        cdef array.array bits_array = array.array('B', bits)
        cdef array.array params_array = array.array('d', [item for sublist in params for item in self._complex_cast(sublist)])
        self.c_class.initialize(bits_array.data.as_uchars, bitCount, params_array.data.as_doubles)

    def multiplexer(self, bits, ctrlCount, params):
        cdef array.array bits_array = array.array('B', bits)

        items = [item for sublist in params for item in sublist] #matrices
        items = [item for sublist in items for item in sublist] #rows
        items = [item for sublist in items for item in self._complex_cast(sublist)] #components
        cdef array.array params_array = array.array('d', items)

        self.c_class.multiplexer(bits_array.data.as_uchars, ctrlCount, params_array.data.as_doubles)

    def reset(self, target):
        self.c_class.reset(target)

    def amplitudes(self):
        return np.array(self.c_class.amplitudes())

    def probabilities(self):
        return np.array(self.c_class.probabilities())

    def measure(self, bits):
        cdef array.array bits_array = array.array('B', bits)
        return self.c_class.measure(bits_array.data.as_uchars, len(bits))

    def measure_all(self):
        return self.c_class.measure_all()

    def measure_shots(self, bits, shots):
        cdef array.array bits_array = array.array('B', bits)
        cppResult = self.c_class.measure_shots(bits_array.data.as_uchars, len(bits), shots)

        result = {}
        for key, value in cppResult:
            result[key] = value

        return result


def qrack_controller_factory():
    """Expose QrackController to Python"""
    cdef PyQrackController py_obj = PyQrackController()
    # Set extension pointer to existing C++ class ptr
    py_obj.c_class = new QrackController()
    return py_obj
